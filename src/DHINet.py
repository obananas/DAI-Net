import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

class DHINet(nn.Module):
    def __init__(self, vocab_size, ddi_adj, MPNNSet, N_fingerprints, average_projection, emb_dim=256, v_dim=64,
                 q_dim=64, h_dim=64, h_out=2,
                 device=torch.device('cpu:0'), act='ReLU', dropout=0.3, k=1):
        super(DHINet, self).__init__()

        self.device = device
        self.k = k
        # pre-embedding
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(2)])
        self.dropout = nn.Dropout(dropout)
        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(2)])
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )
        self.h_out = h_out
        self.v_net = MLP([v_dim, h_dim], act=act, dropout=dropout)
        self.q_net = MLP([q_dim, h_dim], act=act, dropout=dropout)

        # MPNN global embedding
        self.molecule_set = list(zip(*MPNNSet))

        self.drug_emb = MolecularGraphEncoder(N_fingerprints, emb_dim, layer_hidden=2, device=device).forward(
            self.molecule_set)
        self.drug_emb = torch.mm(average_projection.to(device=self.device), self.drug_emb.to(device=self.device))
        self.drug_emb.to(device=self.device)
        self.MPNN_output = nn.Linear(vocab_size[2], vocab_size[2])
        self.MPNN_layernorm = nn.LayerNorm(vocab_size[2])
        self.bn = nn.BatchNorm1d(h_dim)

        self.mlp21 = nn.Sequential(nn.Tanh(),
                                   nn.Linear(h_dim, h_dim))

        self.mlp22 = nn.Sequential(nn.LeakyReLU(0.25),
                                   nn.Linear(h_dim, vocab_size[2]),
                                   nn.LeakyReLU(0.25),
                                   nn.Linear(vocab_size[2], vocab_size[2]),
                                   nn.Sigmoid())

        q = self.drug_emb
        self.sym = nn.Parameter(torch.Tensor(h_dim, 1).normal_())
        self.h_mat = nn.Parameter(torch.Tensor(h_dim, h_out, q.size(0)).normal_())
        self.h_bias = nn.Parameter(torch.Tensor(h_dim, h_out, q.size(0)).normal_())

        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.init_weights()

    def forward(self, input, softmax=False):
        # patient health representation
        i1_seq = []
        i2_seq = []

        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        for adm in input:
            i1 = sum_embedding(
                self.dropout(
                    self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device))))  # (1,1,dim)
            i2 = sum_embedding(
                self.dropout(self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))
            i1_seq.append(i1)
            i2_seq.append(i2)
        i1_seq = torch.cat(i1_seq, dim=1)  # (1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1)  # (1,seq,dim)

        o1, h1 = self.encoders[0](
            i1_seq
        )
        o2, h2 = self.encoders[1](
            i2_seq
        )
        patient_representations = torch.cat([o1, o2], dim=-1).squeeze(dim=0)  # (seq, dim*2)
        query = self.query(patient_representations)[-1:, :]  # (seq, dim)

        v = query
        q = self.drug_emb
        v_num = v.size(0)
        q_num = q.size(0)
        v_ = self.v_net(v)
        q_ = self.q_net(q)
        att_maps = torch.einsum('shq,sk,kq->shq', (self.h_mat, torch.matmul(self.sym, v_), q_.t())) + self.h_bias

        if softmax:
            tmp = nn.functional.sigmoid(att_maps.view(v_num, self.h_out, q_num))
            att_maps = tmp.view(v_num, self.h_out, q_num)
        # aa = att_maps[:, 0, :]
        logits = torch.einsum('vk, kq->vq', (v_, att_maps[:, 0, :]))
        for i in range(1, self.h_out):
            logits_i = torch.einsum('vk, kq->vq', (v_, att_maps[:, i, :]))
            logits += logits_i

        conf_score = torch.einsum('sq, qk->sk', (att_maps[:, 0, :], q_))
        conf_score = self.mlp21(conf_score)
        conf_score = torch.matmul(v_, conf_score)
        for i in range(1, self.h_out):
            logits_i = torch.einsum('sq, qk->sk', (att_maps[:, i, :], q_))
            logits_i = self.mlp21(logits_i)
            logits_i = torch.matmul(v_, logits_i)
            conf_score += logits_i

        conf_score = self.mlp22(conf_score)

        # gama = 1
        result = logits * conf_score

        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)

        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)


class MolecularGraphEncoder(nn.Module):
    def __init__(self, N_fingerprint, dim, layer_hidden, device):
        super(MolecularGraphEncoder, self).__init__()
        self.device = device
        self.embed_fingerprint = nn.Embedding(N_fingerprint, dim).to(self.device)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim).to(self.device)
                                            for _ in range(layer_hidden)])
        self.layer_hidden = layer_hidden

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch proc essing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(self.device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i + m, j:j + n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.mm(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def forward(self, inputs):

        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        """MPNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            # fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.
            fingerprint_vectors = hs

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        return molecular_vectors


class MLP(nn.Module):
    def __init__(self, dims, act='ReLU', dropout=0):
        super(MLP, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))

        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
