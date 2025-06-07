# meta-llama/Llama-3.2-1B
# Qwen/Qwen3-0.6B

data_file=./sample_train_qa_Baichuan2-13B-Base_4096_dataset/data-00000-of-00001.arrow
model_dir=baichuan-inc/Baichuan2-13B-Base
experiment_name=HuatuoGPT2_7B

# accelerate launch --config_file ./training_config/zero.yaml \
#     --num_processes 8 \
#     --num_machines 1 \
#     --machine_rank 0 \
#     --deepspeed_multinode_launcher standard train_huatuo.py \
#     --experiment_name ${experiment_name} \
#     --model_path ${model_dir} \
#     --max_seq_len 4096 \
#     --gradient_accumulation_steps 4 \
#     --data_path ${data_file} \
#     --output_dir ./ckpts \
#     --log_dir ./train_logs \
#     --n_epochs 2 \
#     --warmup_rates 0.01 \
#     --train_bsz_per_gpu 4 \
#     --eval_bsz_per_gpu 4 \
#     --learning_rate 1e-4 \
#     --gradient_checkpointing > training.log 2>&1 &

!cd HuatuoGPT-II-clone; accelerate launch --config_file ./adaption/one_stage_training/training_config/zero.yaml \
    --num_processes 1 \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard adaption/one_stage_training/train_huatuo.py \
    --experiment_name Qwen3_0.6B \
    --model_path Qwen/Qwen3-0.6B \
    --max_seq_len 4096 \
    --gradient_accumulation_steps 1 \
    --data_path ./train_qa_0.01p_en_Baichuan2-13B-Base_4096_dataset \
    --output_dir ./ckpts \
    --log_dir ./train_logs \
    --n_epochs 2 \
    --warmup_rates 0.01 \
    --train_bsz_per_gpu 1 \
    --eval_bsz_per_gpu 1 \
    --learning_rate 1e-4 \
    --gradient_checkpointing > training.log 2>&1

# sed -i 's/\r$//' adaption/one_stage_training/train.sh
