"""Cached model aliases. Importing this module never probes caches or GPUs."""
import argparse
import json


MODEL_REGISTRY = {
    "deepseek-r1-distill-qwen-14b": {"repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"},
    "gemma-3-1b-it": {"repo": "google/gemma-3-1b-it"},
    "gemma-3-4b-it": {"repo": "google/gemma-3-4b-it"},
    "gemma-3-12b-it": {"repo": "google/gemma-3-12b-it"},
    "gemma-3-27b-it": {"repo": "google/gemma-3-27b-it"},
    "gemma-4-31b-it": {"repo": "google/gemma-4-31B-it"},
    "llama-3.1-8b-instruct": {"repo": "meta-llama/Llama-3.1-8B-Instruct"},
    "llama-3.2-1b-instruct": {"repo": "meta-llama/Llama-3.2-1B-Instruct"},
    "llama-3.2-3b-instruct": {"repo": "meta-llama/Llama-3.2-3B-Instruct"},
    # BF16 ~140 GB requires multiple 80 GB GPUs; actual headroom is model/config dependent.
    "llama-3.3-70b-instruct": {"repo": "meta-llama/Llama-3.3-70B-Instruct", "tp_size": 2},
    "mistral-7b-instruct-v0.3": {"repo": "mistralai/Mistral-7B-Instruct-v0.3"},
    "mistral-small-3.1-24b-instruct": {"repo": "mistralai/Mistral-Small-3.1-24B-Instruct-2503"},
    "mistral-large-instruct-2411": {"repo": "mistralai/Mistral-Large-Instruct-2411"},
    "gpt-oss-120b": {"repo": "openai/gpt-oss-120b"},
    "qwen2.5-7b": {"repo": "Qwen/Qwen2.5-7B"},
    "qwen2.5-7b-instruct": {"repo": "Qwen/Qwen2.5-7B-Instruct"},
    "qwen2.5-72b-instruct": {"repo": "Qwen/Qwen2.5-72B-Instruct", "tp_size": 2},
    "qwen3-0.6b": {"repo": "Qwen/Qwen3-0.6B"},
    "qwen3-1.7b": {"repo": "Qwen/Qwen3-1.7B"},
    "qwen3-4b": {"repo": "Qwen/Qwen3-4B"},
    "qwen3-8b": {"repo": "Qwen/Qwen3-8B"},
    "qwen3-14b": {"repo": "Qwen/Qwen3-14B"},
    "qwen3-32b": {"repo": "Qwen/Qwen3-32B", "health_wait_min": 60},
    "sarvam-105b": {"repo": "sarvamai/sarvam-105b"},
    "glm-4.5-air": {"repo": "zai-org/GLM-4.5-Air"},
}


def resolve_model(alias):
    if alias not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model alias {alias!r}. Available: {', '.join(MODEL_REGISTRY)}")
    return {"tp_size": 1, "health_wait_min": 60, "extra_serve_args": "", **MODEL_REGISTRY[alias]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alias", nargs="?")
    parser.add_argument("--field", action="append")
    args = parser.parse_args()
    if args.alias is None:
        print("\n".join(MODEL_REGISTRY))
        return
    try:
        model = resolve_model(args.alias)
        if args.field:
            for field in args.field:
                print(model[field])
        else:
            print(json.dumps(model, indent=2))
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
