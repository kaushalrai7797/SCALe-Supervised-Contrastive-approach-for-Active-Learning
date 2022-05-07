# SCALe: Supervised Contrastive approach for Active Learning

## To reproduce our results please run the following commands

pip install -r requirements.txt

## Training via siamese loss
python experiments.py --dataset "trec6" --acq_fn "random"

## Supervised contrastive learning
python supcon.py
