
# pip install -U sentence-transformers

python generate_emb.py \
  --jsonl /home/lym/MultiPoint/datasets/t2017/train.json \
  --data_dir /home/lym/MultiPoint/datasets/t2017 \
  --split train \
  --hf_model /home/lym/MultiPoint/models/sbert-roberta-large

python generate_emb.py \
  --jsonl /home/lym/MultiPoint/datasets/t2017/val.json \
  --data_dir /home/lym/MultiPoint/datasets/t2017 \
  --split val \
  --hf_model /home/lym/MultiPoint/models/sbert-roberta-large

python generate_emb.py \
  --jsonl /home/lym/MultiPoint/datasets/t2017/test.json \
  --data_dir /home/lym/MultiPoint/datasets/t2017 \
  --split test \
  --hf_model /home/lym/MultiPoint/models/sbert-roberta-large
