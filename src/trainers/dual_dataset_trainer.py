import ast
import contextlib
import json
import shutil
from enum import Enum
from typing import Dict, Optional, List

import math
import os
import subprocess
import time

import numpy as np
import torch
import transformers
import wandb

from data_binding import PrecisionType, TrainArgs, CustomSchedulerType, CustomIntervalStrategy, DataArguments
from torch import autocast, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from warmup_scheduler import GradualWarmupScheduler
from contextlib import nullcontext

from .vigia_trainer import VigiaTrainer, AverageMeter, ProgressMeter


class DualDatasetTrainer(VigiaTrainer):

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 train_dataloader: List[DataLoader],
                 eval_dataloader: List[DataLoader],
                 optimizer: Optimizer = None,
                 args: TrainArgs = None,
                 data_args: DataArguments = None,
                 config: Dict = None,):
        assert len(train_dataloader) == 2, f"DualDatasetTrainer expects 2 datasets but received {len(train_dataloader)}"
        # assert len(train_dataloader[0]) == len(train_dataloader[1]), f"Train dataloader have different sizes: {len(train_dataloader[0])} != {len(train_dataloader[1])}"
        if len(train_dataloader[0]) != len(train_dataloader[1]):
            print(f"Train dataloader have different sizes: {len(train_dataloader[0])} != {len(train_dataloader[1])}, will be limited by the smallest one")
        super().__init__(model, tokenizer, train_dataloader[0], eval_dataloader[0], optimizer, args, data_args, config)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

    def train_loop(self, epoch, callback=None):
        """Main training loop."""
        ngpus_per_node = torch.cuda.device_count()
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        ce_losses = AverageMeter('CeLoss', ':.4e')
        cont_losses = AverageMeter('ContLoss', ':.4e')

        world_size = dist.get_world_size() if dist.is_initialized() else 1

        progress = ProgressMeter(
            self.batches_per_epoch,
            [batch_time, losses, ce_losses, cont_losses],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        self.model.train()
        skip_steps = self.args.resume_from_step % self.batches_per_epoch if self.args.resume_from_checkpoint else 0

        dtype = torch.bfloat16 # Your desired dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        end = time.time()

        for i, (batch1, batch2) in enumerate(zip(self.train_dataloader[0], self.train_dataloader[1])):
            with torch.set_grad_enabled(True):
                if self.global_step < skip_steps:
                    if (i + 1) % self.args.gradient_accumulation_steps == 0:
                        self.global_step += 1
                    continue

            global_batch_size = batch1['input_ids'].size(0) + batch2['input_ids'].size(0)

            loss = 0

            is_final_accumulation_step = (i + 1) % self.args.gradient_accumulation_steps == 0 or (i == self.batches_per_epoch - 1)
            context1 = self.model.no_sync()
            context3 =  autocast(device_type=device.type, dtype=dtype, enabled=self.scaler.is_enabled())
            with context1, context3, torch.autograd.graph.allow_mutation_on_saved_tensors():
                model_mode_1 = [m[0] for m in batch1.pop("supported_tasks", [])][0]
                batch1['mode'] = model_mode_1
                local_batch_size_1 = batch1['input_ids'].size(0)

                # Perform forward pass
                outputs1, task_loss1 = self.training_step(batch1)

                # Scale the loss for gradient accumulation.
                # Each micro-batch's loss contributes 1/N to the total gradient.
                scaled_loss1 = task_loss1 / self.args.gradient_accumulation_steps

                # Perform backward pass for Task 1. Gradients are computed locally.
                # DDP synchronization is skipped due to no_sync().
                if self.scaler_dtype == torch.float32 or self.args.load_dtype in [PrecisionType.FP16, PrecisionType.BF16]:
                    scaled_loss1.backward()
                else:
                    self.scaler.scale(scaled_loss2).backward()

                # Log metrics (use original unscaled loss)
                if model_mode_1 in ['captioning', 'textgen']:
                    ce_losses.update(task_loss1.item(), local_batch_size_1)
                elif model_mode_1 == 'retrieval':
                    cont_losses.update(task_loss1.item(), local_batch_size_1)
                else:
                    raise NotImplementedError(f"Unsupported model mode: {model_mode_1}")

            # --- Process Batch 2 (Task 2) ---
            # Determine context for the second batch's backward pass:
            # - If this is the final accumulation step ('i'), we NEED DDP sync, so use nullcontext.
            # - Otherwise, it's not the final step, so skip DDP sync using no_sync().
            context2 = contextlib.nullcontext() if is_final_accumulation_step else self.model.no_sync()
            with context2, context3, torch.autograd.graph.allow_mutation_on_saved_tensors():
                model_mode_2 = [m[0] for m in batch2.pop("supported_tasks", [])][0]
                batch2['mode'] = model_mode_2
                local_batch_size_2 = batch2['input_ids'].size(0)

                # Perform forward pass
                outputs2, task_loss2 = self.training_step(batch2)

                # Scale the loss for gradient accumulation
                scaled_loss2 = task_loss2 / self.args.gradient_accumulation_steps

                # Perform backward pass for Task 2.
                # If context2 is nullcontext (i.e., is_final_accumulation_step is True),
                # this backward call WILL trigger DDP synchronization for ALL gradients
                # accumulated since the last optimizer.step() (from batch1 and batch2 in this step 'i',
                # plus potentially gradients from previous 'i' steps in this cycle).
                # If context2 is no_sync, gradients are computed locally only.
                if self.scaler_dtype == torch.float32 or self.args.load_dtype in [PrecisionType.FP16, PrecisionType.BF16]:
                    scaled_loss2.backward()
                else:
                    self.scaler.scale(scaled_loss2).backward()

                # Log metrics (use original unscaled loss)
                if model_mode_2 in ['captioning', 'textgen']:
                    ce_losses.update(task_loss2.item(), local_batch_size_2)
                elif model_mode_2 == 'retrieval':
                    cont_losses.update(task_loss2.item(), local_batch_size_2)
                else:
                    raise NotImplementedError(f"Unsupported model mode: {model_mode_2}")

             # --- Logging Average Loss for Step 'i' ---
            # Calculate average loss across tasks for this step 'i' for logging purposes
            # Note: Ensure global_batch_size calculation is correct if batch sizes differ.
            avg_loss_step_i = (task_loss1.item() + task_loss2.item()) / 2
            losses.update(avg_loss_step_i, global_batch_size) # Log average loss for this step

            # Update weights
            if ((i + 1) % self.args.gradient_accumulation_steps == 0) or (i == self.batches_per_epoch - 1):
                self.global_step += 1
                self.gradient_accumulation()
                self.scheduler.step()
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

            self.norm_grads()


            if ((((i + 1) % self.args.gradient_accumulation_steps == 0) or (i == self.batches_per_epoch - 1)) and
                    (self.global_step == 1 or self.global_step % self.args.logging_steps == 0)):

                ex_per_sec = (self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * world_size) / batch_time.avg

                if self.is_main_process():
                    progress.display(self.global_step % self.batches_per_epoch)

                log_dict = {
                    'loss': losses.avg,
                    'ce_loss': ce_losses.avg,
                    'contrastive_loss': cont_losses.avg,
                    'total_secs_per_batch': batch_time.avg,
                    'examples_per_sec': ex_per_sec,
                    # 'lr': self.scheduler.get_last_lr()[0],
                }
                for i, param_group in enumerate(self.optimizer.param_groups):
                    log_dict[f'lr_{param_group["name"]}'] = param_group['lr']


                if self.args.report_to_wandb:
                    log_dict = {f"train/{k}": v for k, v in log_dict.items()}
                    wandb.log(log_dict, step=self.global_step)

                batch_time.reset()
                losses.reset()
                ce_losses.reset()
                cont_losses.reset()

                if self.args.save_strategy == CustomIntervalStrategy.STEPS and self.global_step % self.args.save_steps == 0:
                    self.print(f"--> saving model@{self.global_step}")
                    ckpt_name = self.save_checkpoint()
                    if self.args.infer_checkpoints and self.global_step >= self.args.warmup_before_inference and self.is_main_process():
                        # we use sbatch(SLURM) to run the inference script, change with whatever you use
                        subprocess.call(['sbatch', f'scripts/{self.args.infer_file}', os.path.join(self.args.output_dir, ckpt_name), self.eval_file])

                if self.args.evaluation_strategy == CustomIntervalStrategy.STEPS and self.global_step % self.args.eval_steps == 0:
                    self.print(f"--> running validation@{self.global_step}")
                    self.model.eval()
                    eval_score = self.validation_loop(f"checkpoint_{self.global_step}")
                    if self.args.save_strategy == CustomIntervalStrategy.BEST_EVAL and self.best_val_loss > eval_score:
                        self.best_val_loss = eval_score
                        self.print(f"----> saving best model@{self.global_step} with weighted recall: {self.best_val_loss}")
                        self.save_checkpoint()
                    self.model.train()

            if self.global_step >= self.total_steps:
                break


    def training_step(self, batch):

        model_mode = batch.pop("mode")

        for key in batch.keys():
            batch[key] = batch[key].to(self.device)

        # compute output
        concat_captions = np.random.uniform(0, 1) < 0.5
        concat_captions = concat_captions and model_mode == 'captioning'

        (model_output, full_labels, last_embedding, _, visual_embs, task_loss) = self.model(
            **batch, mode=model_mode, concat_captions=concat_captions)
        output = model_output.logits

        return output, task_loss

    def gradient_accumulation(self):
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        if model.model.config.freeze_lm and not model.model.config.freeze_ret:
            # Zero out gradients of the embedding matrix outside of [RET].
            for param in model.model.input_embeddings.parameters():
                assert param.grad.shape[0] == len(self.tokenizer)
                # Keep other embeddings frozen.
                mask = torch.arange(param.grad.shape[0]) != model.model.ret_token_id
                param.grad[mask, :] = 0

            if not model.model.lm.config.tie_word_embeddings:
                # do the same for the output embeddings
                for param in model.model.lm.get_output_embeddings().parameters():
                    if param.grad is not None:
                        assert param.grad.shape[0] == len(self.tokenizer)
                        mask = torch.arange(param.grad.shape[0]) != model.model.ret_token_id
                        try:
                            param.grad[mask, :] = 0
                        except:
                            param.grad[mask] = 0

        # compute gradient and do SGD step
        if self.args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
        if self.scaler_dtype == torch.float32 or self.args.load_dtype in [PrecisionType.FP16, PrecisionType.BF16]: # cant do mixed precision with 16bit inputs
            self.optimizer.step()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        self.optimizer.zero_grad()

    def norm_grads(self):
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        if model.model.config.freeze_lm and not model.model.config.freeze_ret:
            with torch.no_grad():
                # Normalize trainable embeddings.
                frozen_norm = torch.norm(model.model.input_embeddings.weight[:-1, :], dim=1).mean(0)
                trainable_weight = model.model.input_embeddings.weight[-1, :]
                model.model.input_embeddings.weight[-1, :].div_(torch.norm(trainable_weight) / frozen_norm)

                if not model.model.lm.config.tie_word_embeddings:
                    # do the same for the output embeddings
                    frozen_norm = torch.norm(model.model.lm.get_output_embeddings().weight[:-1, :], dim=1).mean(0)
                    trainable_weight = model.model.lm.get_output_embeddings().weight[-1, :]
                    model.model.lm.get_output_embeddings().weight[-1, :].div_(
                        torch.norm(trainable_weight) / frozen_norm)