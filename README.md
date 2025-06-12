Citation: [HUatuoGPT-II](https://github.com/FreedomIntelligence/HuatuoGPT-II)

### Prepare


### Preprocess
- data_path: Output of data_prepare.py
- train_bsz_per_gpu: Max num of conversation per input_ids
- log_gap_per_loader: Log wandb every N batches

python adaption/one_stage_training/data_process.py \
    --data_path all_data/huatuo_trained_1p.json \
    --model_path Qwen/Qwen3-0.6B \
    --max_seq_len 512 \
    --train_bsz_per_gpu 32 \
    --log_gap_per_loader 1 \
    --num_workers 1

### Train
- data_path: Output of data_process.py

accelerate launch --config_file ./adaption/one_stage_training/training_config/zero.yaml \
    --num_processes 2 \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard adaption/one_stage_training/train_huatuo.py \
    --experiment_name Qwen3_0.6B \
    --model_path Qwen/Qwen3-0.6B \
    --max_seq_len 512 \
    --gradient_accumulation_steps 1 \
    --data_path ./all_data/huatuo_trained_1p_Qwen3-0.6B_512_dataset \
    --output_dir ./ckpts \
    --log_dir ./train_logs \
    --n_epochs 1 \
    --warmup_rates 0.01 \
    --train_bsz_per_gpu 1 \
    --eval_bsz_per_gpu 1 \
    --learning_rate 1e-4 \
    --gradient_checkpointing > training.log 2>&1

### Inference
python upload_to_hf.py
    --save_dir ./ckpts/Qwen3_0.6B/checkpoint-0-59 \

