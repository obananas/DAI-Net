# DAI-Net: Dual Adaptive Interaction Network for Coordinated Medication Recommendation
<!-- **DAI-Net: Dual Adaptive Interaction Network for Coordinated Drug Recommendation** -->
This is the official repo for Dual Adaptive Interaction Network (DAI-Net), a simple method for coordinated drug recommendation. 

<div style='display:flex; gap: 0.25rem; '>
<a href='LICENCE'><img src='https://img.shields.io/badge/License-Apache 2.0-g.svg'></a>
<a href='https://doi.org/10.1109/JBHI.2024.3425833'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
<a href='https://zhuanlan.zhihu.com/p/714258819'><img src='https://img.shields.io/badge/zhihu-Markdown-blue'></a>
</div>

## 🔥 Update
* [2024-7-30]: ⭐️ Paper of DAI-Net online (accepted by IEEE JBHI). Check out [this link](https://doi.org/10.1109/JBHI.2024.3425833) for details.
* [2023-11-29]: 🚀🚀 Codes released.

## 🕹️ Usage
### Requirements
```python
torch == 1.8.0+cu111
torch-geometric == 1.0.3
torch-scatter == 2.0.9
torch-sparse == 0.6.12
```

### Package Dependency

- first, install the [`rdkit`](https://www.rdkit.org/) (RDKit: Open-Source Cheminformatics Software) conda environment.

```python
conda create -c conda-forge -n DAI-Net  rdkit
conda activate DAI-Net

# can also use the following in your current env
pip install rdkit-pypi
```

- then, in DAI-Net environment, install the following package
```python
pip install scikit-learn, dill, dnc
```
Note that torch setup may vary according to GPU hardware. Generally, run the following
```python
pip install torch
```
If you are using RTX 3090, then plase use the following, which is the right way to make torch work.
```python
python3 -m pip install --user torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

- Finally, install other packages if necessary
```python
pip install [xxx] # any required package if necessary, maybe do not specify the version, the packages should be compatible with rdkit
```

Here is a list of reference versions for all package

```shell
pandas: 1.3.0
dill: 0.3.4
torch: 1.8.0+cu111
rdkit: 2021.03.4
scikit-learn: 0.24.2
numpy: 1.21.1
```
Let us know any of the package dependency issue. Please pay special attention to pandas, some report that a high version of pandas would raise error for dill loading.

### Processing

- Go to https://physionet.org/content/mimiciii/1.4/ to download the MIMIC-III dataset (You may need to get the certificate)

```bash
  wget -r -N -c -np --user [account] --ask-password https://physionet.org/files/mimiciii/1.4/
  ```

- Go into the folder and unzip required three files and copy them to the `~/cs598dl4h-project/data/input/` folder

```bash
  cd ~/physionet.org/files/mimiciii/1.4
  gzip -d PROCEDURES_ICD.csv.gz # procedure information
  gzip -d PRESCRIPTIONS.csv.gz  # prescription information
  gzip -d DIAGNOSES_ICD.csv.gz  # diagnosis information
  cp PROCEDURES_ICD.csv PRESCRIPTIONS.csv DIAGNOSES_ICD.csv ~/cs598dl4h-project/data/input/
  ```

- Download additional files in the `~/cs598dl4h-project/data/input/` folder

```bash
  cd ~/cs598dl4h-project/data/input/
  ./get_additional_files.sh
  ```

- Processing the data to get a complete `records_final.pkl`

  ```bash
  cd ~/cs598dl4h-project/data
  python processing.py
  ```
  
### 📌 Project Structure
- `data/`
  - `processing.py`: The data preprocessing file.
- `input/`
    - `PRESCRIPTIONS.csv`: the prescription file from MIMIC-III raw dataset
- `DIAGNOSES_ICD.csv`: the diagnosis file from MIMIC-III raw dataset
- `PROCEDURES_ICD.csv`: the procedure file from MIMIC-III raw dataset
- `RXCUI2atc4.csv`: this is a NDC-RXCUI-ATC4 mapping file, and we only need the RXCUI to ATC4 mapping. This file is obtained from https://github.com/ycq091044/SafeDrug.
- `drug-atc.csv`: this is a CID-ATC file, which gives the mapping from CID code to detailed ATC code (we will use the prefix of the ATC code latter for aggregation). This file is obtained from https://github.com/ycq091044/SafeDrug.
- `rxnorm2RXCUI.txt`: rxnorm to RXCUI mapping file. This file is obtained from https://github.com/ycq091044/SafeDrug.
- `drugbank_drugs_info.csv`: drug information table downloaded from drugbank here https://www.dropbox.com/s/angoirabxurjljh/drugbank_drugs_info.csv?dl=0, which is used to map drug name to drug SMILES string.
- `drug-DDI.csv`: this a large file, containing the drug DDI information, coded by CID. The file could be downloaded from https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing
  - `output/`
    - `atc3toSMILES.pkl`: drug ID (we use ATC-3 level code to represent drug ID) to drug SMILES string dict
- `ddi_A_final.pkl`: ddi adjacency matrix
- `ehr_adj_final.pkl`: used in GAMENet baseline (if two drugs appear in one set, then they are connected)
- `records_final.pkl`: The final diagnosis-procedure-medication EHR records of each patient, used for train/val/test split.
- `voc_final.pkl`: diag/prod/med index to code dictionary
- `src/`
  - `SafeDrug.py`: our model
- baseline models:
- `GAMENet.py`
    - `DMNC.py`
    - `Leap.py`
    - `Retain.py`
    - `ECC.py`
    - `LR.py`
  - setting file
- `model.py`
    - `util.py`
    - `layer.py`
  - analysis file
- `Result-Analysis.ipynb`
- `dependency.sh`
- `requirements.txt`
- `README.md`

After the processing have been done, we get the following statistics:

```bash
# patients  6350
# clinical events  15032
# diagnosis  1958
# med  112
# procedure 1430
# avg of diagnoses  10.5089143161256
# avg of medicines  11.647751463544438
# avg of procedures  3.8436668440659925
# avg of vists  2.367244094488189
# max of diagnoses  128
# max of medicines  64
# max of procedures  50
# max of visit  29
```

### Process Data
The processed data is in the path
```python
\DAI-Net\data
```
You can also process data with
- MIMIC-III
```python
python processing.py
```
- MIMIC-IV
```python
python processing_4.py
```
### Run the code

```python
python DAI-Net.py
```

here is the argument:

    usage: DAI-Net.py [-h] [--Test] [--model_name MODEL_NAME]
                   [--resume_path RESUME_PATH] [--lr LR]
                   [--target_ddi TARGET_DDI] [--kp KP] [--dim DIM]
    
    optional arguments:
      -h, --help            show this help message and exit
      --Test                test mode
      --model_name MODEL_NAME
                            model name
      --resume_path RESUME_PATH
                            resume path
      --lr LR               learning rate
      --target_ddi TARGET_DDI
                            target ddi
      --kp KP               coefficient of P signal
      --dim DIM             dimension

## 📑 Citation
If you find our project useful, we hope you can star our repo and cite our paper as follows:
```
@article{zou2024dai,
  title={DAI-Net: Dual Adaptive Interaction Network for Coordinated Medication Recommendation},
  author={Zou, Xin and He, Xiao and Zheng, Xiao and Zhang, Wei and Chen, Jiajia and Tang, Chang},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2024},
  publisher={IEEE}
}
```

## Credits

Our work followed the original codes at https://github.com/ycq091044/SafeDrug.
