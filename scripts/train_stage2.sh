#!/bin/bash

# Stage 2: Visual Instruction Tuning
# End-to-end training on general visual instruction datasets to build strong
# image understanding capabilities. Uses LoRA for the LLM and VE.

export STAGE3=0

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

HF_HUB_OFFLINE=1 torchrun --nnodes 1 --nproc_per_node 4 --master_port $MASTER_PORT src/main.py \
    --run_name vigia_stage2 \
    --project_name vigia_stage2 \
    --text_decoder 'meta-llama/Llama-3.1-8B-Instruct' \
    --visual_encoder "google/siglip-so400m-patch14-224" \
    --ckpt_path /path/to/your/experiments/vigia_stage1/checkpoint_19500/ \
    --resume_from_checkpoint False \
    --output_dir /path/to/your/experiments/vigia_stage2 \
    --overwrite_output_dir True \
    --dual_dataset True \
    --data_path data_config/stage2_cap.yaml \
    --dataset_type multi_dataset \
    --dataset_name laion400m \
    --second_data_path data_config/stage2_ret.yaml \
    --second_dataset_type multi_dataset \
    --second_dataset_name laion400m \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_second_train_batch_size 6 \
    --per_device_eval_batch_size 4 \
    --shuffle_data True \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy no \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 5000 \
    --max_steps 15000 \
    --save_total_limit 10 \
    --warmup_steps 100 \
    --dedicated_optimizers True \
    --llm_lr 0.00005 \
    --llm_weight_decay 0.01 \
    --ve_lr 0.00005 \
    --ve_weight_decay 0.0 \
    --cap_layers_lr 0.0005 \
    --cap_layers_weight_decay 0.03 \
    --ret_layers_lr 0.001 \
    --ret_layers_weight_decay 0.03 \
    --weight_decay 0.03 \
    --warmup_ratio 0.03 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type constant \
    --logging_steps 1 \
    --seq_max_length 2048 \
    --perpetual False \
    --report_to_wandb True \
    --infer_checkpoints False \
    --infer_file "infer_checkpoint.sh" \
    --load_dtype BF16 \
    --load_in_8bits False \
    --mixed_precision BF16 \
    --warmup_before_inference 0 \
    --reload_optimizer False \
    --parallel_type DP \
    --use_half False \
    --lora True \
    --lora_alpha 128 \
    --lora_rank 32 \
    --lora_dropout 0.03 \
    --debug False \
    --dataset_kwargs "{'supported_tasks': ['captioning', 'retrieval']}" \
    --use_pos_emb True \
    --freeze_lm False \
    --freeze_vm False \
    --freeze_cap False \
    --freeze_ret False \
    --n_visual_tokens 1 \
    --image_embed_dropout_prob 0.0 \
    --shared_emb_dim 512 \
    --text_embed_dropout_prob 0.1 \
    --ret_loss_scale 1.0 \
    --cap_loss_scale 1.0 \
    --use_cls_token False \
    --connector_type mlp \
    --projector_type linear
