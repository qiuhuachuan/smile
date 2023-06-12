import argparse
import ujson
from tqdm import tqdm


def format_example(example: dict) -> dict:
    context = example['input']
    target = example["output"]
    return {"context": context, "target": target}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",
                        type=str,
                        default="train_data/train.json")
    parser.add_argument("--save_path",
                        type=str,
                        default="train_data/train.jsonl")

    args = parser.parse_args()
    examples = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        examples += ujson.load(f)
        print(len(examples))

    with open(args.save_path, 'w', encoding='utf-8') as f:
        for example in tqdm(examples, desc="formatting.."):
            json_string = ujson.dumps(format_example(example),
                                      ensure_ascii=False)
            f.write(json_string + '\n')


if __name__ == "__main__":
    main()
    print('convert to jsonl -> DONE')
