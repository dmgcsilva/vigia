#!/bin/bash

# Stage 1: Initialization
# Bridges the modality gap between the visual encoder and the language model.
# Both backbones are frozen; only the MLP connector and retrieval projectors are trained.

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

HF_HUB_OFFLINE=1 torchrun --nnodes 1 --nproc_per_node 4 --master_port $MASTER_PORT src/main.py \
    --run_name vigia_stage1 \
    --project_name vigia_stage1 \
    --text_decoder 'meta-llama/Llama-3.1-8B-Instruct' \
    --visual_encoder "google/siglip-so400m-patch14-224" \
    --resume_from_checkpoint False \
    --output_dir /path/to/your/experiments/vigia_stage1 \
    --overwrite_output_dir True \
    --data_path /path/to/your/data/laion400m/laion400m_unified_10m.parquet \
    --dual_dataset True \
    --dataset_type multimode_dataset \
    --dataset_name laion400m \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy no \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 5000 \
    --max_steps 19500 \
    --save_total_limit 10 \
    --warmup_steps 100 \
    --dedicated_optimizers True \
    --llm_lr 0.001 \
    --llm_weight_decay 0.03 \
    --ve_lr 0.0005 \
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
    --seq_max_length 100 \
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
    --lora False \
    --lora_alpha 8 \
    --lora_rank 4 \
    --lora_dropout 0.1 \
    --debug False \
    --dataset_kwargs "{'supported_tasks': ['captioning', 'retrieval']}" \
    --use_pos_emb True \
    --freeze_lm True \
    --freeze_vm True \
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
