import argparse
import random
import time
import warnings

import torch
from model.model_firefly import FireFlyConfig, FireFlyForCausalLM
from trainer.train_util import get_model_params, setup_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

warnings.filterwarnings("ignore")


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)

    if args.load_from == "model":
        model = FireFlyForCausalLM(
            FireFlyConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                inference_rope_scaling=bool(args.inference_rope_scaling),
            )
        )
        checkpoint_path = f"./{args.save_dir}/{args.weight}_{args.hidden_size}.pth"
        try:
            model.load_state_dict(
                torch.load(checkpoint_path, map_location=args.device), strict=True
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint structure does not match current model config. "
                "Use the same hidden_size/num_hidden_layers as training "
                "(pretrain/full_sft defaults are hidden_size=768, num_hidden_layers=12)."
            ) from exc
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.load_from,
            trust_remote_code=True,
        )

    get_model_params(model)
    model = model.eval().to(args.device)
    if args.device.startswith("cuda"):
        model = model.half()
    return model, tokenizer


def build_input_text(tokenizer, conversation, prompt, is_pretrain):
    try:
        return tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
    except (AttributeError, TypeError):
        # Some tokenizers only support a minimal chat template signature.
        try:
            return tokenizer.apply_chat_template(conversation, tokenize=False)
        except (AttributeError, TypeError):
            if is_pretrain:
                return (
                    f"{tokenizer.bos_token}user\n{prompt}{tokenizer.eos_token}\n"
                    f"{tokenizer.bos_token}assistant\n"
                )
            return tokenizer.bos_token + prompt


def main():
    parser = argparse.ArgumentParser(description="FireFly LLM evaluation")
    parser.add_argument(
        "--load_from",
        default="model",
        type=str,
        help="Model path. Use 'model' for local FireFly checkpoint, otherwise a Hugging Face path.",
    )
    parser.add_argument(
        "--save_dir",
        default="out",
        type=str,
        help="Directory that stores local checkpoint files.",
    )
    parser.add_argument(
        "--weight",
        default="full_sft",
        type=str,
        help="Checkpoint name prefix for local FireFly model.",
    )
    parser.add_argument(
        "--hidden_size",
        default=768,
        type=int,
        help="Hidden size of local FireFly model.",
    )
    parser.add_argument(
        "--num_hidden_layers",
        default=12,
        type=int,
        help="Number of transformer layers in local FireFly model.",
    )
    parser.add_argument(
        "--inference_rope_scaling",
        default=0,
        type=int,
        choices=[0, 1],
        help="Enable YaRN rope scaling for local model (0/1).",
    )
    parser.add_argument(
        "--max_new_tokens",
        default=512,
        type=int,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        default=0.85,
        type=float,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        default=0.95,
        type=float,
        help="Top-p nucleus sampling value.",
    )
    parser.add_argument(
        "--history_turns",
        default=0,
        type=int,
        help="How many latest chat messages to keep as context. 0 means no history.",
    )
    parser.add_argument(
        "--show_speed",
        default=1,
        type=int,
        choices=[0, 1],
        help="Whether to print generation speed in tokens/s (0/1).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
        help="Inference device, for example: cuda or cpu.",
    )
    args = parser.parse_args()

    auto_prompts = [
        "你有什么特长？",
        "What are your core strengths?",
        "为什么天空是蓝色的？",
        "Why is the sky blue?",
        "请用Python写一个计算斐波那契数列的函数。",
        "Write a Python function to compute the Fibonacci sequence.",
        "解释一下光合作用的基本过程。",
        "Explain the basic process of photosynthesis.",
        "如果明天下雨，我应该如何准备出门？",
        "If it rains tomorrow, how should I prepare before going out?",
        "比较一下猫和狗作为宠物的优缺点。",
        "Compare the pros and cons of having cats vs dogs as pets.",
    ]

    conversation = []
    model, tokenizer = init_model(args)

    input_mode = int(input("[0] Auto test\n[1] Manual input\n"))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    prompt_iter = auto_prompts if input_mode == 0 else iter(lambda: input("User: "), "")

    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0:
            print(f"User: {prompt}")

        conversation = conversation[-args.history_turns :] if args.history_turns else []
        conversation.append({"role": "user", "content": prompt})

        input_text = build_input_text(
            tokenizer,
            conversation,
            prompt,
            is_pretrain=("pretrain" in args.weight),
        )
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(
            args.device
        )

        print("Assistant: ", end="")
        start_time = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=1.0,
        )

        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]) :],
            skip_special_tokens=True,
        )
        conversation.append({"role": "assistant", "content": response})

        if args.show_speed:
            generated_token_count = len(generated_ids[0]) - len(inputs["input_ids"][0])
            elapsed = max(time.time() - start_time, 1e-6)
            print(f"\n[Speed]: {generated_token_count / elapsed:.2f} tokens/s\n")
        else:
            print("\n")


if __name__ == "__main__":
    main()
