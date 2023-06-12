import argparse
import json
from tqdm import tqdm

import datasets
from transformers import AutoTokenizer, AutoConfig


def preprocess(tokenizer, config, example, max_seq_length):
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer.encode(prompt,
                                  max_length=max_seq_length,
                                  truncation=True)
    target_ids = tokenizer.encode(target,
                                  max_length=max_seq_length,
                                  truncation=True,
                                  add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


def read_jsonl(path, max_seq_length, skip_overlength=False):
    model_name = "THUDM/chatglm-6b"
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name,
                                        trust_remote_code=True,
                                        device_map='auto')

    with open(path, "r", encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            feature = preprocess(tokenizer, config, example, max_seq_length)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            yield feature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path",
                        type=str,
                        default="train_data/train.jsonl")
    parser.add_argument("--save_path", type=str, default="train_data/smile")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--skip_overlength", action='store_true')
    args = parser.parse_args()

    dataset = datasets.Dataset.from_generator(lambda: read_jsonl(
        args.jsonl_path, args.max_seq_length, args.skip_overlength))
    dataset.save_to_disk(args.save_path)


if __name__ == "__main__":
    main()
    print('tokenize -> DONE')
