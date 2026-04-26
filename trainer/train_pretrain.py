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

from FireFly.fireflyoptim import FireFlyProb
from FireFly.bitLinear import collect_bitlinear_modules

from itertools import islice

warnings.filterwarnings("ignore")


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
            loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            eta_time = spend_time / max(1, step - start_step) * (iters - step) // 60
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"loss: {loss.item():.4f}, lr: {lr:.8f}, eta: {eta_time:.1f}min"
            )
            if swanlab is not None:
                swanlab.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "train/eta_min": eta_time,
                    },
                    step=epoch * iters + step,
                )

        if step % args.save_interval == 0:
            model.eval()
            save_model_weight(model, args)
            lm_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                save_dir=args.checkpoint_dir,
                lm_config=args.lm_config,
                weight=args.save_weight,
            )
            model.train()

        del input_ids, labels, outputs, loss

    if last_step > 0:
        model.eval()
        save_model_weight(model, args)
        lm_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=last_step,
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
        "--epochs", type=int, default=2, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Training batch size per GPU"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=2000, help="Number of warmup steps"
    )
    parser.add_argument(
        "--base_ratio",
        type=float,
        default=0.02,
        help="Base ratio for FireFly optimizer (between 0 and 1)",
    )
    parser.add_argument(
        "--vote_interval", type=int, default=32, help="Interval for FireFly voting"
    )
    parser.add_argument(
        "--vote_threshold", type=float, default=24, help="Threshold for FireFly voting"
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
        default=768,
        help="Hidden size of the model",
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=12,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--max_seq_length",
        default=340,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../dataset/pretrain_t2t.jsonl",
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

    setup_seed(42)

    os.makedirs(args.save_dir, exist_ok=True)
    args.checkpoint_dir = "checkpoints"
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    lm_config = FireFlyConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
    )
    args.lm_config = lm_config
    ckp_data = (
        lm_checkpoint(
            lm_config, weight=args.save_weight, save_dir=args.checkpoint_dir
        )
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
        save_dir=args.save_dir,
        device=args.device,
    )
    train_ds = PretrainDataset(
        args.data_path, tokenizer, max_length=args.max_seq_length
    )
    optimizer = FireFlyProb(
        model.parameters(),
        lr_dense=args.learning_rate,
        base_ratio=args.base_ratio,
        vote_interval=args.vote_interval,
        vote_threshold=args.vote_threshold,
        clip_grad=args.clip_grad,
        bit_modules=collect_bitlinear_modules(model),
    )

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        start_epoch = ckp_data.get("epoch", 0)
        start_step = ckp_data.get("step", 0)

    for epoch in range(start_epoch, args.epochs):
        setup_seed(42 + epoch)
        loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        if epoch == start_epoch and start_step > 0:
            Logger(f"resume: skip first {start_step} batches")
            loader = islice(loader, start_step, None)

        train_epoch(
            epoch,
            loader,
            math.ceil(len(train_ds) / args.batch_size),
            start_step,
            swanlab=swanlab if args.use_swanlab else None,
            args=args,
            optimizer=optimizer,
            model=model,
        )


if __name__ == "__main__":
    main()
