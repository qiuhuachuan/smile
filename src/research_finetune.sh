export WANDB_PROJECT=research_smile
nohup python -u finetune.py \
    --dataset_path train_data/smile_full \
    --lora_rank 8 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 10 \
    --report_to wandb \
    --output_dir research_smile_epoch2_rank8_full > ./log/research_smile_epoch2_rank8_full.log &