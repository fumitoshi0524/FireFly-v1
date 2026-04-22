import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random
import math
import numpy as np
import torch

from transformers import AutoTokenizer
from model.model_firefly import FireFlyForCausalLM


def get_model_params(model):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    Logger(f"Model Params: {total:.2f}M")


def Logger(content):
    print(content)


# add warmup
def get_lr(current_step, total_steps, lr, warmup_steps=2000):
    if current_step < warmup_steps:
        return lr * (0.1 + 0.9 * current_step / warmup_steps)
    t = (current_step - warmup_steps) / (total_steps - warmup_steps)
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * t)))


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def lm_checkpoint(
    lm_config,
    weight="full_sft",
    model=None,
    optimizer=None,
    epoch=0,
    step=0,
    wandb=None,
    save_dir="../checkpoints",
    **kwargs,
):
    os.makedirs(save_dir, exist_ok=True)
    ckp_path = f"{save_dir}/{weight}_{lm_config.hidden_size}.pth"
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}_resume.pth"

    if model is not None:
        state_dict = model.state_dict()
        state_dict = {k: v.cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + ".tmp"
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, "get_run"):
                run = wandb.get_run()
                wandb_id = getattr(run, "id", None) if run else None
            else:
                wandb_id = getattr(wandb, "id", None)

        resume_data = {
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "wandb_id": wandb_id,
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, "state_dict"):
                    resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + ".tmp"
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location="cpu")
            return ckp_data
        return None


def init_model(
    lm_config,
    from_weight="pretrain",
    tokenizer_path="../model",
    save_dir="../out",
    device="cuda",
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = FireFlyForCausalLM(lm_config)

    if from_weight != "none":
        weight_path = f"{save_dir}/{from_weight}_{lm_config.hidden_size}.pth"
        if os.path.exists(weight_path):
            weights = torch.load(weight_path, map_location=device)
            model.load_state_dict(weights, strict=False)

    get_model_params(model)
    Logger(
        f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M"
    )
    return model.to(device), tokenizer
