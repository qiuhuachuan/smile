import ujson
import os

def get_all_sessions():
    source_dir = f'./data'
    
    all_sessions = []
    context = []
    counter = 0
    for idx in range(56032):
        try:
            with open(f'{source_dir}/{idx}.json', 'r', encoding='utf-8') as f1:
                data = ujson.load(f1)

            for j, item in enumerate(data):
                context.append(item)
                if item['role'] == 'counselor':
                    if j == 0:
                        pass
                    else:
                        dialogue_history_list = context[-10:-1]
                        new_item = {
                            'idx': counter,
                            'dialogue_history_list': dialogue_history_list,
                            'response': item['content']
                        }
                        all_sessions.append(new_item)
                        counter += 1
        except Exception as e:
            pass
    print(f'# training samples: {counter}')
    print('done')
    return all_sessions

def get_dialogue_history(dialogue_history_list: list):
    
    dialogue_history_tmp = []
    for item in dialogue_history_list:
        if item['role'] == 'counselor':
            text = '咨询师：'+ item['content']
        else:
            text = '来访者：'+ item['content']
        dialogue_history_tmp.append(text)

    dialogue_history = '\n'.join(dialogue_history_tmp)
    
    return dialogue_history + '\n' + '咨询师：'
    
    

def get_instruction_data(dialogue_history):
    instruction = f'''现在你扮演一位专业的心理咨询师，你具备丰富的心理学和心理健康知识。你擅长运用多种心理咨询技巧，例如认知行为疗法原则、动机访谈技巧和解决问题导向的短期疗法。以温暖亲切的语气，展现出共情和对来访者感受的深刻理解。以自然的方式与来访者进行对话，避免过长或过短的回应，确保回应流畅且类似人类的对话。提供深层次的指导和洞察，使用具体的心理概念和例子帮助来访者更深入地探索思想和感受。避免教导式的回应，更注重共情和尊重来访者的感受。根据来访者的反馈调整回应，确保回应贴合来访者的情境和需求。请为以下的对话生成一个回复。

对话：
{dialogue_history}'''

    return instruction


if __name__ == '__main__':
    all_sessions = get_all_sessions()
    train_dir = './train_dir'
    os.makedirs(train_dir, exist_ok=True)
    
    with open(f'{train_dir}/train.json', 'w', encoding='utf-8') as f:
        for item in all_sessions:
            dialogue_history_list = item['dialogue_history_list']
            dialogue_history = get_dialogue_history(dialogue_history_list=dialogue_history_list)

            output = item['response']

            instruction = get_instruction_data(
                dialogue_history=dialogue_history
            )
            text = ujson.dumps({'instruction': instruction, 'output': output}, ensure_ascii=False)
            f.write(text+'\n')
            
    print('********')
    