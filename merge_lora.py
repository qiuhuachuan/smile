import torch
from model import MODE
import argparse
from peft import PeftModel


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_model_dir', default='THUDM/chatglm2-6b', type=str, help='')
    # 你的lora模型保存位置，根据情况修改
    parser.add_argument('--model_dir', default="output-glm2-psyqa/epoch-2-step-5602", type=str, help='')
    parser.add_argument('--mode', default="glm2", type=str, help='')

    return parser.parse_args()


if __name__ == '__main__':
    args = set_args()
    base_model = MODE[args.mode]["model"].from_pretrained(args.ori_model_dir, torch_dtype=torch.float16)
    lora_model = PeftModel.from_pretrained(base_model, args.model_dir, torch_dtype=torch.float16)
    lora_model.to("cpu")
    model = lora_model.merge_and_unload()
    MODE[args.mode]["model"].save_pretrained(model, args.model_dir, max_shard_size="2GB")
