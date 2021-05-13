#!/bin/sh
module load Python/3.8.2-GCCcore-9.3.0
pip install torch
pip install torchvision
pip install numpy
pip install seqlogo
python --version
python cnn_filter15_fewerfilters.py --len 9, fewer filters, 100 epochs