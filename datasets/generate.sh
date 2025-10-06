
# pip install -U sentence-transformers

python generate_emb.py \
  --jsonl /home/lym/MultiPoint/datasets/mvsa-s/train.json \
  --data_dir /home/lym/MultiPoint/datasets/mvsa-s \
  --split train \
  --hf_model /home/lym/MultiPoint/models/sbert-roberta-large

python generate_emb.py \
  --jsonl /home/lym/MultiPoint/datasets/mvsa-s/val.json \
  --data_dir /home/lym/MultiPoint/datasets/mvsa-s \
  --split val \
  --hf_model /home/lym/MultiPoint/models/sbert-roberta-large

python generate_emb.py \
  --jsonl /home/lym/MultiPoint/datasets/mvsa-s/test.json \
  --data_dir /home/lym/MultiPoint/datasets/mvsa-s \
  --split test \
  --hf_model /home/lym/MultiPoint/models/sbert-roberta-large
