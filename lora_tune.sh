training_type=smile

CUDA_VISIBLE_DEVICES=0,1,2,3
# THUDM/chatglm2-6b
nohup deepspeed --include=localhost:0,1,2,3 --master_port 8888 train.py \
            --train_path train_dir/train.json \
            --model_name_or_path THUDM/chatglm2-6b \
            --per_device_train_batch_size 1 \
            --max_len 8192 \
            --max_src_len 8192 \
            --learning_rate 1e-4 \
            --weight_decay 0.1 \
            --num_train_epochs 2 \
            --gradient_accumulation_steps 4 \
            --warmup_ratio 0.1 \
            --mode glm2 \
            --train_type lora \
            --lora_dim 16 \
            --lora_alpha 64 \
            --lora_dropout 0.1 \
            --lora_module_name "query_key_value,dense_h_to_4h,dense_4h_to_h,dense" \
            --seed 1234 \
            --ds_file ds_zero2_no_offload.json \
            --gradient_checkpointing \
            --show_loss_step 10 \
            --output_dir ./output-glm2-${training_type} > ./log/${training_type}.log &