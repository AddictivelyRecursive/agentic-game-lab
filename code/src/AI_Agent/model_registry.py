"""Cached model aliases. Importing this module never probes caches or GPUs."""
import argparse
import json


MODEL_REGISTRY = {'llama-3.1-8b-instruct': {'repo': 'meta-llama/Llama-3.1-8B-Instruct'},
 'qwen3-8b': {'repo': 'Qwen/Qwen3-8B'},
 'gemma-3-12b-it': {'repo': 'google/gemma-3-12b-it'},
 'mistral-small-3.1-24b-instruct': {'repo': 'mistralai/Mistral-Small-3.1-24B-Instruct-2503'},
 'deepseek-r1-distill-qwen-14b': {'repo': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'},
 'gpt-oss-120b': {'repo': 'openai/gpt-oss-120b'}}


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
