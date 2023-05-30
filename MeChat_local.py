import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import torch
from transformers import AutoTokenizer, AutoModel

from peft import PeftModel

model = AutoModel.from_pretrained('THUDM/chatglm-6b',
                                  revision='v0.1.0',
                                  trust_remote_code=True)
LaRA_PATH = 'qiuhuachuan/MeChat'
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

        true_out_text = tokenizer.decode(out[0][input_length:])

        answer = true_out_text.replace('\nEND', '').strip()
        return answer


data = []
while True:
    seeker_msg = input('求助者：')
    data.append({'owner': 'seeker', 'msg': seeker_msg})
    input_str = ''
    for item in data:
        if item['owner'] == 'seeker':
            input_str += '求助者：' + item['msg']
        else:
            input_str += '支持者：' + item['msg']
    input_str += '支持者：'
    while len(input_str) > 2000:
        if input_str.index('求助者：') > input_str.index('支持者：'):
            start_idx = input_str.index('求助者：')
        else:
            start_idx = input_str.index('支持者：')
        input_str = input_str[start_idx:]

    wrapped_data = {'input': input_str}

    response = generate_response(data=wrapped_data)
    print(f'支持者：{response}')
    supporter_msg = {'owner': 'supporter', 'msg': response}
    data.append(supporter_msg)
