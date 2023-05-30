import os
import ujson

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import uvicorn
import torch
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from peft import PeftModel

model = AutoModel.from_pretrained('THUDM/chatglm-6b',
                                  revision='v0.1.0',
                                  trust_remote_code=True)
LaRA_PATH = 'qiuhuachuan/MeChat'
model = PeftModel.from_pretrained(model, LaRA_PATH)
model = model.float().to(device='cuda')

tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b',
                                          trust_remote_code=True)


class ChatInfo(BaseModel):
    owner: str
    msg: str
    unique_id: str


app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=['*'],
                   allow_credentials=True,
                   allow_methods=['*'],
                   allow_headers=['*'])


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

        return answer


@app.post('/v1/chat')
async def chat(ChatInfo: ChatInfo):
    unique_id = ChatInfo.unique_id
    existing_files = os.listdir('./dialogues')
    # print(existing_files)
    target_file = f'{unique_id}.json'
    if target_file in existing_files:
        with open(f'./dialogues/{unique_id}.json', 'r', encoding='utf-8') as f:
            data: list = ujson.load(f)
    else:
        data = []
    data.append({
        'owner': ChatInfo.owner,
        'msg': ChatInfo.msg,
        'unique_id': ChatInfo.unique_id
    })
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
    supporter_msg = {
        'owner': 'supporter',
        'msg': response,
        'unique_id': unique_id
    }
    data.append(supporter_msg)
    with open(f'./dialogues/{unique_id}.json', 'w', encoding='utf-8') as f:
        ujson.dump(data, f, ensure_ascii=False, indent=2)
    return {'item': supporter_msg, 'responseCode': 200}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)