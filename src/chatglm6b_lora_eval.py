import os
import ujson

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import torch
from transformers import AutoTokenizer, AutoModel

from peft import PeftModel

model = AutoModel.from_pretrained('THUDM/chatglm-6b',
                                  revision='v0.1.0',
                                  trust_remote_code=True)
LaRA_PATH = 'research_smile_epoch2_rank8_full'
model = PeftModel.from_pretrained(model, LaRA_PATH)
model = model.float().to(device='cuda')

tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b',
                                          trust_remote_code=True)


def format_example(example: dict) -> dict:
    context = f'''Input: {example['input']}\n'''

    return {'context': context, 'target': ''}


def generate_response(data: dict):
    with torch.no_grad():
        feature = format_example(data)
        input_text = feature['context']
        ids = tokenizer.encode(input_text)
        input_length = len(ids)
        input_ids = torch.LongTensor([ids]).to(device='cuda')

        out = model.generate(input_ids=input_ids,
                             max_length=2040,
                             do_sample=True,
                             temperature=0.9,
                             top_p=0.9)

        raw_out_text = tokenizer.decode(out[0])
        print(raw_out_text)
        true_out_text = tokenizer.decode(out[0][input_length:])

        answer = true_out_text.replace('\nEND', '').strip()
        answer = answer.replace(',', '，').replace(':', '：').replace('?', '？')

        return answer


def chat():
    filenames = ['0']
    for filename in filenames:
        for round in ['round1']:
            target_dir = f'./human_eval/{round}/{filename}'
            existing_files = os.listdir(target_dir)
            existing_files = sorted(existing_files)
            for each_file in existing_files:
                print(each_file)
                with open(f'{target_dir}/{each_file}', 'r',
                          encoding='utf-8') as f:
                    data = ujson.load(f)

                ctx = data['ctx']
                input_str = ''
                input_str += ''.join(ctx)
                input_str += '支持者：'

                wrapped_data = {'input': input_str}

                response = generate_response(data=wrapped_data)

                data['chatglm6b_lora_response'] = response
                with open(f'./human_eval/full_epoch2_rank8/{each_file}',
                          'w',
                          encoding='utf-8') as f:
                    ujson.dump(data, f, ensure_ascii=False, indent=2)


chat()
print('done')

# nohup python -u chatglm6b_lora_eval.py > chatglm6b_lora_eval.log &