from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="BlankZ/ragen-checkpoint-step-1200-bf16",
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()

    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")

    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    print(f"tokenizer loaded in {time.perf_counter() - start:.1f}s")

    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="auto",
        max_memory={0: "3.4GiB", "cpu": "48GiB"} if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    print(f"model loaded in {time.perf_counter() - start:.1f}s")
    print(f"device_map={getattr(model, 'hf_device_map', None)}")
    first_device = next(model.parameters()).device
    print(f"first_parameter_device={first_device}")
    if torch.cuda.is_available():
        print(
            "cuda_memory="
            f"{torch.cuda.memory_allocated() / 1024**3:.2f}GiB allocated, "
            f"{torch.cuda.memory_reserved() / 1024**3:.2f}GiB reserved"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an agent in a grid puzzle. Return exactly one action "
                "inside <answer>...</answer>. Valid actions: up, down, left, right."
            ),
        },
        {
            "role": "user",
            "content": (
                "State:\n"
                "#####\n"
                "#A..#\n"
                "#.B.#\n"
                "#..G#\n"
                "#####\n"
                "A is the agent. B is a box. G is the goal. Choose one action."
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(first_device) for key, value in inputs.items()}

    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start
    generated = output_ids[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=False)
    print(f"generated {generated.numel()} tokens in {elapsed:.1f}s")
    print("=== generation ===")
    print(text)


if __name__ == "__main__":
    main()
