
# pip install -U sentence-transformers

python generate_emb.py \
  --jsonl /home/lym/MultiPoint/datasets/tumemo/train.json \
  --data_dir /home/lym/MultiPoint/datasets/tumemo \
  --split train \
  --hf_model /home/lym/MultiPoint/models/sbert-roberta-large

python generate_emb.py \
  --jsonl /home/lym/MultiPoint/datasets/tumemo/val.json \
  --data_dir /home/lym/MultiPoint/datasets/tumemo \
  --split val \
  --hf_model /home/lym/MultiPoint/models/sbert-roberta-large

python generate_emb.py \
  --jsonl /home/lym/MultiPoint/datasets/tumemo/test.json \
  --data_dir /home/lym/MultiPoint/datasets/tumemo \
  --split test \
  --hf_model /home/lym/MultiPoint/models/sbert-roberta-large
