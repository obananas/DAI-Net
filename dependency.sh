#!/bin/bash

CUDA=$1

pip install -r requirements.txt

if [ -z "$CUDA" ]
then
    pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
else
    pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
fi
