# Required environment variables:
# TAG: tag for the trail
# TYPE: finetune / prompt / prompt-demo  
# TASK: SST-2 / sst-5 / mr / cr / mpqa / subj / trec / CoLA / MNLI / SNLI / QNLI / RTE / MRPC / QQP / STS-B
# BS: batch size (recommendation: 2 / 4 / 8)
# LR: learning rate (recommendation: 1e-5 / 2e-5 / 5e-5)
# SEED: random seed (13 / 21 / 42 / 87 / 100)
# MODEL: pre-trained model name (roberta-*, bert-*), see Transformers model list
export TOKENIZERS_PARALLELISM=false
# Number of training instances per label
TASK=Mul_T2015_Contrastive_Fusion_Add_Caption
MODEL=/home/lym/MultiPoint/models/roberta-large
TYPE=prompt-demo
TAG=######################################################################\n
# Training steps
MAX_STEP=500

# Validation steps
EVAL_STEP=100

TEXT_MAX_LENGTH=512
ADD_IMAGE=True
EVALUATE_DURING_TRAINING=True
IMAGE_MODEL_NAME=nf_resnet50
# Task specific parameters
# The default length is 128 and the default number of samples is 16.
# For some tasks, we use longer length or double demo (when using demonstrations, double the maximum length).
# For some tasks, we use smaller number of samples to save time (because of the large size of the test sets).
# All those parameters are set arbitrarily by observing the data distributions.
IMAGE_TOKEN="<image>_0"
PROMPT_TOKEN="<prompt>"

# TEMPLATE1=*cls*$IMAGE_TOKEN*is*caption_0*sep+**sent_0*_It_was*mask*.*sep+* 
# TEMPLATE2=*cls*$IMAGE_TOKEN*is*caption_0*sep+*_The_sentense_\"*sent_0*\"_has*mask*_sentiment*sep+* 
# TEMPLATE3=*cls*$IMAGE_TOKEN*is*caption_0*sep+*$PROMPT_TOKEN*mask*$PROMPT_TOKEN*sent_0*$PROMPT_TOKEN*sep+*

# TEMPLATE4=*cls*$IMAGE_TOKEN*is*caption_0*sep+*Text_:_\"*sent_0*\"_.*_sentiment_of_text_:*mask*.*sep+*

TEMPLATE1=*cls*$IMAGE_TOKEN*is*caption_0*sep+**sent_0**It_was*mask**about**aspect_0*.*sep+*
TEMPLATE2=*cls*$IMAGE_TOKEN*is*caption_0*sep+**_The_sentense_*"*sent_0*"*has**mask***_sentiment**about**aspect_0*.*sep+*
TEMPLATE3=*cls*$IMAGE_TOKEN*is*caption_0*sep+*$PROMPT_TOKEN*mask*$PROMPT_TOKEN*sent_0*$PROMPT_TOKEN*aspect_0*$PROMPT_TOKEN*sep+*
TEMPLATE4=*cls*$IMAGE_TOKEN*is*caption_0*sep+*Text*:*"*sent_0*"*.*Aspect*:*aspect_0*.*_sentiment_of_text_:*mask*.*sep+*

MAPPING="{'negative':'terrible','neutral':'okay','positive':'great'}"
TASK_EXTRA="--max_seq_len 512"

# Gradient accumulation steps
# For medium-sized GPUs (e.g., 2080ti with 10GB memory), they can only take 
# a maximum batch size of 2 when using large-size models. So we use gradient
# accumulation steps to achieve the same effect of larger batch sizes.
REAL_BS=2
# GS=$(expr $BS / $REAL_BS)
GS=1

# Use a random number to distinguish different trails (avoid accidental overwriting)
template_type='[3-4]'

for percent in 1 
do 
    for num_sample in 1
    do
        for NUM_IMAGE_TOKENS in 1
        do
            for num_prompt_tokens in 2 4 6
            do
                for train_batch_size in 8
                do
                    for lr in 3e-6 4e-6 5e-6 7e-6 8e-6 1e-5 1.2e-5 1.4e-5 1.6e-5 1.8e-5 2e-5 3e-5 4e-5 5e-5 7e-5 
                    do
                        for SEED in 13 #21 42 87 100
                        do
                            CUDA_VISIBLE_DEVICES=5 python run_new_add_caption.py \
                            --task_name $TASK \
                            --data_dir /home/lym/MultiPoint/datasets/t2015 \
                            --add_image $ADD_IMAGE \
                            --image_model_name $IMAGE_MODEL_NAME \
                            --num_image_tokens $NUM_IMAGE_TOKENS \
                            --num_prompt_tokens $num_prompt_tokens \
                            --do_train \
                            --do_eval \
                            --do_predict \
                            --demo_filter True \
                            --demo_filter_model sbert-roberta-large \
                            --evaluate_during_training \
                            --model_name_or_path $MODEL \
                            --few_shot_type $TYPE \
                            --max_seq_length $TEXT_MAX_LENGTH \
                            --per_device_train_batch_size $train_batch_size  \
                            --per_device_eval_batch_size 32 \
                            --gradient_accumulation_steps $GS \
                            --learning_rate $lr \
                            --max_steps $MAX_STEP \
                            --logging_steps $EVAL_STEP \
                            --eval_steps $EVAL_STEP \
                            --output_dir output/$TASK/$TYPE-$percent-Template_type-$template_type-$SEED-$MODEL-lr-$lr-batch_size-$train_batch_size-num_image_tokens-$NUM_IMAGE_TOKENS  \
                            --log_dir log_new/$TASK/$TYPE-$percent-Template_type-$template_type-$SEED-$MODEL-lr-$lr-batch_size-$train_batch_size-num_image_tokens-$NUM_IMAGE_TOKENS  \
                            --seed $SEED \
                            --tag $TAG \
                            --template $TEMPLATE3 \
                            --template2 $TEMPLATE4 \
                            --mapping $MAPPING \
                            --percent $percent \
                            --num_sample $num_sample \
                            $TASK_EXTRA \
                            $1 

                            # Delete the checkpoint 
                            # Since we need to run multiple trials, saving all the checkpoints takes 
                            # a lot of storage space. You can find all evaluation results in `log` file anyway.
                            rm -r output/$TASK/$TYPE-$percent-Template_type-$template_type-$SEED-$MODEL-lr-$lr-batch_size-$train_batch_size-num_image_tokens-$NUM_IMAGE_TOKENS
                        done
                    done
                done
            done
        done
    done
done