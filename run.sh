#!/bin/bash -l

#$ -N lomo
#$ -j y
#$ -l gpus=1
#$ -l gpu_c=8.0

cd /projectnb/devorlab/bcraus/AnalysisCode/predictNE
source venv/bin/activate

python -m crossvalidation.lomo