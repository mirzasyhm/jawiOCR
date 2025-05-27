#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import summarize

from strhub.data.module import SceneTextDataModule
from strhub.models.base import BaseSystem
from strhub.models.utils import get_pretrained_weights

# Copied from OneCycleLR
def _annealing_cos(start, end, pct):
    'Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0.'
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out


def get_swa_lr_factor(warmup_pct, swa_epoch_start, div_factor=25, final_div_factor=1e4) -> float:
    """Get the SWA LR factor for the given `swa_epoch_start`. Assumes OneCycleLR Scheduler."""
    total_steps = 1000
    start_step = int(total_steps * warmup_pct) - 1
    end_step = total_steps - 1
    step_num = int(total_steps * swa_epoch_start) - 1
    pct = (step_num - start_step) / (end_step - start_step)
    return _annealing_cos(1, 1 / (div_factor * final_div_factor), pct)


class PrintMetricsCallback(Callback):
    """Callback to print batch & epoch metrics including loss, accuracy, learning rate."""

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        avg_loss = metrics.get('train_loss') or metrics.get('loss') or 0.0
        avg_acc = metrics.get('train_accuracy') or metrics.get('train_acc') or metrics.get('val_accuracy') or 0.0
        optim = trainer.optimizers[0]
        lr = optim.param_groups[0].get('lr', None)
        print(f"[Epoch {epoch}] avg_loss={float(avg_loss):.4f} accuracy={float(avg_acc):.2f} lr={lr:.6e}")

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        val_acc = metrics.get('val_accuracy', None)
        if val_acc is None:
            print(f"[Epoch {epoch}] val_accuracy=N/A")
        else:
            try:
                v = float(val_acc)
            except Exception:
                v = val_acc
            print(f"[Epoch {epoch}] val_accuracy={v:.2f}")


@hydra.main(config_path='configs', config_name='main', version_base='1.2')
def main(config: DictConfig):
    trainer_strategy = 'auto'
    with open_dict(config):
        config.data.root_dir = hydra.utils.to_absolute_path(config.data.root_dir)
        gpu = config.trainer.get('accelerator') == 'gpu'
        devices = config.trainer.get('devices', 0)
        if gpu:
            config.trainer.precision = 'bf16-mixed' if torch.get_autocast_gpu_dtype() is torch.bfloat16 else '16-mixed'
        if gpu and devices > 1:
            trainer_strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
            config.trainer.val_check_interval //= devices
            if config.trainer.get('max_steps', -1) > 0:
                config.trainer.max_steps //= devices

    if config.model.get('perm_mirrored', False):
        assert config.model.perm_num % 2 == 0, 'perm_num should be even if perm_mirrored = True'

    model: BaseSystem = hydra.utils.instantiate(config.model)
    if config.pretrained is not None:
        m = model.model if config.model._target_.endswith('PARSeq') else model
        m.load_state_dict(get_pretrained_weights(config.pretrained))
    print(summarize(model, max_depth=2))

    datamodule: SceneTextDataModule = hydra.utils.instantiate(config.data)

    checkpoint = ModelCheckpoint(
        monitor='val_accuracy',
        mode='max',
        save_top_k=3,
        save_last=True,
        filename='{epoch}-{step}-{val_accuracy:.4f}-{val_NED:.4f}',
    )
    swa_epoch_start = 0.75
    swa_lr = config.model.lr * get_swa_lr_factor(config.model.warmup_pct, swa_epoch_start)
    swa = StochasticWeightAveraging(swa_lr, swa_epoch_start)

    metrics_cb = PrintMetricsCallback()

    cwd = (
        HydraConfig.get().runtime.output_dir
        if config.ckpt_path is None
        else str(Path(config.ckpt_path).parents[1].absolute())
    )
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=TensorBoardLogger(cwd, '', '.'),
        strategy=trainer_strategy,
        enable_model_summary=False,
        callbacks=[metrics_cb, checkpoint, swa],
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)


if __name__ == '__main__':
    main()
