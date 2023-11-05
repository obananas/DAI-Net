# DHINet
Dual-path Hierarchical Interaction Network for Coordinated Drug Recommendation

This is an implementation of our model DHINet and the baselines in the paper. 
<hr>

## Requirements
```python
torch == 1.8.0+cu111
torch-geometric == 1.0.3
torch-scatter == 2.0.9
torch-sparse == 0.6.12
```
## Package Dependency

- first, install the rdkit conda environment
```python
conda create -c conda-forge -n DHINet  rdkit
conda activate DHINet

# can also use the following in your current env
pip install rdkit-pypi
```

- then, in DHINet environment, install the following package
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

## Process Data
The processed data is in the path
```python
\DHINet\data
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
## Run the code

```python
python DHINet.py
```

here is the argument:

    usage: DHINet.py [-h] [--Test] [--model_name MODEL_NAME]
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
