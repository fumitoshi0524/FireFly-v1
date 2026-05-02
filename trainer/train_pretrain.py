import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import math
import time
import warnings
import torch

from torch.utils.data import DataLoader
from model.model_firefly import FireFlyConfig
from dataset.llm_dataset import PretrainDataset
from trainer.train_util import (
    get_lr,
    Logger,
    lm_checkpoint,
    setup_seed,
    init_model,
)

from FireFly.fireflyoptim import FireFlyOptim
from FireFly.bitLinear import collect_bitlinear_modules

from itertools import islice

warnings.filterwarnings("ignore")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_project_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def save_model_weight(model, args):
    weight_path = f"{args.save_dir}/{args.save_weight}_{args.lm_config.hidden_size}.pth"
    weight_tmp = weight_path + ".tmp"
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, weight_tmp)
    os.replace(weight_tmp, weight_path)
    del state_dict


def train_epoch(
    epoch,
    loader,
    iters,
    start_step=0,
    swanlab=None,
    args=None,
    optimizer=None,
    model=None,
):

    start_time = time.time()
    last_step = 0
    accumulation_steps = max(1, int(args.accumulation_steps))
    optimizer.zero_grad(set_to_none=True)

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        last_step = step
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        lr = get_lr(
            epoch * iters + step,
            args.epochs * iters,
            args.learning_rate,
            warmup_steps=args.warmup_steps,
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with torch.amp.autocast(device_type=args.device, dtype=torch.bfloat16):
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss / accumulation_steps

        loss.backward()
        local_step = step - start_step
        should_step = (local_step % accumulation_steps == 0) or (step == iters)
        if should_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            eta_time = spend_time / max(1, step - start_step) * (iters - step) // 60
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"loss: {loss.item() * accumulation_steps:.4f}, lr: {lr:.8f}, eta: {eta_time:.1f}min"
            )
            if swanlab is not None:
                swanlab.log(
                    {
                        "train/loss": loss.item() * accumulation_steps,
                        "train/lr": lr,
                        "train/eta_min": eta_time,
                    },
                    step=epoch * iters + step,
                )

        if step % args.save_interval == 0:
            model.eval()
            save_model_weight(model, args)
            resume_epoch = epoch + 1 if step >= iters else epoch
            resume_step = 0 if step >= iters else step
            lm_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=resume_epoch,
                step=resume_step,
                save_dir=args.checkpoint_dir,
                lm_config=args.lm_config,
                weight=args.save_weight,
            )
            model.train()

        del input_ids, labels, outputs, loss

    if last_step > 0:
        model.eval()
        save_model_weight(model, args)
        resume_epoch = epoch + 1 if last_step >= iters else epoch
        resume_step = 0 if last_step >= iters else last_step
        lm_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=resume_epoch,
            step=resume_step,
            save_dir=args.checkpoint_dir,
            lm_config=args.lm_config,
            weight=args.save_weight,
        )
        model.train()


def main():
    parser = argparse.ArgumentParser(description="FireFly Pretrain")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="out",
        help="Checkpoint save directory",
    )
    parser.add_argument(
        "--save_weight", type=str, default="pretrain", help="Checkpoint weight name"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size per GPU"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=8000, help="Number of warmup steps"
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps. 0 means disabled.",
    )
    parser.add_argument(
        "--lr_int8",
        type=float,
        default=1e-5,
        help="Learning rate for integer-step updates on INT8 weights.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-3,
        help="Weight decay for weight_scale parameters to prevent explosion.",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.0,
        help="OU mean-reversion rate for the residual accumulator (0 = disabled).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for training (e.g., 'cuda' or 'cpu')",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type for training (e.g., 'float32' or 'bfloat16')",
    )
    parser.add_argument(
        "--clip_grad", type=float, default=1.0, help="Max norm for gradient clipping"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Interval for logging training status",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1000,
        help="Interval for saving model checkpoints",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=1536,
        help="Hidden size of the model",
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=20,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/pretrain_t2t_mini.jsonl",
        help="Path to the training data",
    )
    parser.add_argument(
        "--from_weight",
        type=str,
        default="none",
        help="Pretrained weight name to initialize from",
    )
    parser.add_argument(
        "--from_resume",
        default=0,
        type=int,
        choices=[0, 1],
        help="Whether to resume training from checkpoint (1 for yes, 0 for no)",
    )
    parser.add_argument(
        "--use_swanlab",
        action="store_true",
        help="Whether to use SwanLab for experiment tracking",
    )
    parser.add_argument(
        "--swanlab_project",
        type=str,
        default="FireFly-Pretrain",
        help="SwanLab project name",
    )

    args = parser.parse_args()
    args.accumulation_steps = max(1, int(args.accumulation_steps))

    setup_seed(42)

    args.save_dir = resolve_project_path(args.save_dir)
    args.data_path = resolve_project_path(args.data_path)
    os.makedirs(args.save_dir, exist_ok=True)
    args.checkpoint_dir = resolve_project_path("checkpoints")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    lm_config = FireFlyConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
    )
    args.lm_config = lm_config
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir=args.checkpoint_dir)
        if args.from_resume == 1
        else None
    )

    if args.use_swanlab:
        import swanlab

        swanlab.init(
            project=args.swanlab_project,
            name=f"FireFly-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}",
            id=ckp_data.get("wandb_id") if ckp_data else None,
            resume="must" if ckp_data else None,
        )

    model, tokenizer = init_model(
        lm_config,
        from_weight=args.from_weight,
        tokenizer_path=resolve_project_path("model"),
        save_dir=args.save_dir,
        device=args.device,
    )
    train_ds = PretrainDataset(
        args.data_path, tokenizer, max_length=args.max_seq_length
    )
    optimizer = FireFlyOptim(
        model.parameters(),
        lr_dense=args.learning_rate,
        lr_int8=args.lr_int8,
        clip_grad=args.clip_grad,
        theta=args.theta,
        weight_decay=args.weight_decay,
        bit_modules=collect_bitlinear_modules(model),
    )

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        start_epoch = int(ckp_data.get("epoch", 0))
        start_step = int(ckp_data.get("step", 0))

    iters_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    legacy_iters_per_epoch = len(train_ds) // args.batch_size
    has_partial_last_batch = (len(train_ds) % args.batch_size) != 0
    if has_partial_last_batch and start_step == legacy_iters_per_epoch:
        start_epoch += 1
        start_step = 0
        Logger("resume: migrated legacy epoch-end checkpoint to next epoch")
    if start_step >= iters_per_epoch:
        start_epoch += start_step // iters_per_epoch
        start_step = start_step % iters_per_epoch
    if start_epoch >= args.epochs:
        if ckp_data:
            save_model_weight(model, args)
            Logger("resume: exported final weight to save_dir")
        Logger("resume: checkpoint already reached target epochs, nothing to train")
        return

    for epoch in range(start_epoch, args.epochs):
        epoch_start_step = start_step if epoch == start_epoch else 0
        setup_seed(42 + epoch)
        loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        if epoch_start_step > 0:
            Logger(f"resume: skip first {epoch_start_step} batches")
            loader = islice(loader, epoch_start_step, None)

        train_epoch(
            epoch,
            loader,
            iters_per_epoch,
            epoch_start_step,
            swanlab=swanlab if args.use_swanlab else None,
            args=args,
            optimizer=optimizer,
            model=model,
        )


if __name__ == "__main__":
    main()
