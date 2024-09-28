<p align="center">
  <img src="./image/psychologist.png" width=100px/>
</p>

# 中文心理健康支持对话 · 数据集(SmileChat)与大模型(MeChat)

## 🎉🎉🎉 accepted to the EMNLP 2024 Findings

<img src="https://img.shields.io/badge/Version-1.0-brightgreen" /> <img src="https://img.shields.io/badge/python-3.8+-blue.svg" /> <a href='https://arxiv.org/pdf/2305.00450.pdf'><img src='https://img.shields.io/badge/ArXiv-2305.00450-red'></a>

## 模型地址

https://huggingface.co/qiuhuachuan/MeChat

## Release
**欢迎大家关注我的其他优秀的同类工作**

- 🔥🔥🔥 [2024/8/27] https://github.com/qiuhuachuan/interactive-agents

- 🔥🔥🔥 [2023/11/30] https://github.com/qiuhuachuan/PsyChat

## 项目简介

_For more details, see our paper:_ [smile paper](https://arxiv.org/pdf/2305.00450.pdf 'smile paper')

**MeChat** (**Me**ntal Health Support **Chat**bot)

🎉🎉🎉 **一个更加强大的心理健康对话模型 PsyChat，请参考此仓库**：https://github.com/qiuhuachuan/PsyChat

**背景**：我们都知道心理健康的重要性以及心理健康问题一直是我们关注的焦点。开发用于心理健康支持的专业化对话系统引起了学术界的巨大关注。

**动机**：事实上，建立一个实用、安全、有效的心理健康对话智能体是许多研究人员一直追求的目标。然而，创建这样一个系统的第一步就是要有训练数据。

**挑战**：收集并发布这一类高质量的、真实的大规模数据来促进这一领域的发展面对诸多挑战。首先是数据隐私保护的问题、其次是收集数据所耗费的大量时间与各种成本（平台搭建、真实的受试者与专业的支持者的招聘、筛选、管理等）。

**研究意义**：由大语言模型驱动的虚拟咨询师，作为一种用于心理健康的创新解决思路，可以有效地解决获得性障碍，如高昂的治疗费用、训练有素的专业人员的短缺。此外，该对话系统可以为有需要的人提供有效且实用的在线咨询，能够保护用户隐私，减轻在求助过程中的耻感。

**方法**：我们提出了 SMILE (Single-turn to Multi-turn Inclusive Language Expansion)，一种单轮对话到多轮对话的包容性语言扩展技术。具体来说，利用 ChatGPT 将单轮长对话转换为多轮对话，更好地模拟了真实世界中求助者与支持者之间的多轮对话交流。

**结果**：我们首先对语言转换进行分析，相比其他基线方法，验证了我们提出方法的可行性。其次，我们完成了对话多样性的研究，包括词汇特征、语义特征和对话主题，阐明我们所提方法的有效性。再者，我们通过专家评估，证明了所提方法生成数据的质量高于其他基线方法。因此，我们利用此方法进行大规模数据生成，构建了一个约 55k 的多轮对话数据集。最后，为了更好的评估该数据集的质量，我们利用此数据集训练了一个用于心理健康支持的聊天机器人。在真实数据集的自动化评估和人类与对话系统的交互评估，结果均表明对话系统在心理健康支持能力得到显著提升，进一步证实所生成的数据集具备高质量和实用性的特性。

**未来展望**：利用生成的数据来训练模型，并用于心理健康支持是一个不错的选择。但我们注意到，现有生成数据的对话轮数较短，与真实咨询数据的策略分布上存在一定的差距。因此，秉持让用户受益的原则，需要重点关注模型安全性能，包括自杀干预、敏感信息应对和避免错误信息等，我们任重道远。

本项目开源的**中文心理健康支持模型**由 ChatGLM2-6B LoRA 指令微调得到。数据集通过扩展**真实的心理互助 QA**为多轮的心理健康支持多轮对话，提高了通用语言大模型**在心理健康支持领域的表现**，更加符合在长程多轮对话的应用场景。

## 快速体验

- 术语说明

  - client (来访者) == help-seeker (求助者)

  - counselor (咨询师) == supporter (支持者)

1. 配置环境

```bash
pip install -r requirements.txt
```

2. 运行交互文件 MeChat.py（要求单卡显存 >= 20G）

```bash
python MeChat.py
```

```Python
from transformers import AutoTokenizer, AutoModel

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

def get_instruction(dialogue_history):
    instruction = f'''现在你扮演一位专业的心理咨询师，你具备丰富的心理学和心理健康知识。你擅长运用多种心理咨询技巧，例如认知行为疗法原则、动机访谈技巧和解决问题导向的短期疗法。以温暖亲切的语气，展现出共情和对来访者感受的深刻理解。以自然的方式与来访者进行对话，避免过长或过短的回应，确保回应流畅且类似人类的对话。提供深层次的指导和洞察，使用具体的心理概念和例子帮助来访者更深入地探索思想和感受。避免教导式的回应，更注重共情和尊重来访者的感受。根据来访者的反馈调整回应，确保回应贴合来访者的情境和需求。请为以下的对话生成一个回复。

对话：
{dialogue_history}'''

    return instruction


tokenizer = AutoTokenizer.from_pretrained('qiuhuachuan/MeChat', trust_remote_code=True)
model = AutoModel.from_pretrained('qiuhuachuan/MeChat', trust_remote_code=True).half().cuda()
model = model.eval()

dialogue_history_list = []
while True:
    usr_msg = input('来访者：')
    if usr_msg == '0':
        exit()
    else:
        dialogue_history_list.append({
            'role': 'client',
            'content': usr_msg
        })
        dialogue_history = get_dialogue_history(dialogue_history_list=dialogue_history_list)
        instruction = get_instruction(dialogue_history=dialogue_history)
        response, history = model.chat(tokenizer, instruction, history=[], temperature=0.8, top_p=0.8)
        print(f'咨询师：{response}')
        dialogue_history_list.append({
            'role': 'counselor',
            'content': response
        })
```

以下是一个交互示例：

```txt
来访者：你好
咨询师：你好，很高兴能够帮助到你。你可以告诉我更多关于你的情况吗？
来访者：我最近睡眠很差怎么办？
咨询师：最近睡眠质量不好，你有没有尝试过一些方法来改善睡眠呢？
来访者：我可能最近晚上睡觉前总喜欢玩手机，越玩人就越兴奋
咨询师：晚上玩手机确实会刺激大脑，导致入睡困难。你可以尝试在睡觉前设定一个手机使用时间，逐渐减少玩手机的时间，以帮助放松入睡。
来访者：我应该设置一个晚上9点的闹钟，强制我不玩手机，让我的大脑开始放松。
咨询师：是的，设置一个晚上的闹钟是一个很好的方法。你可以尝试在9点之前放下手机，让自己进入放松的状态，然后逐渐入睡。
来访者：我 尝 试了一下，看看能否改善？
咨询师：很好，试试看吧。另外，你还可以尝试一些放松的技巧，比如深呼吸、冥想或者听一些轻柔的音乐来帮助你放松身心，帮助入睡。
来访者：非常感谢你的建议，我会去试试的
咨询师：不客气，希望这些方法能对你有所帮助。如果还有其他问题，随时都可以来找我倾诉。
```

## 模型训练

### 数据转换

`data` 目录下包含使用 `smile` 方法构建的所有对话。详情可参考论文细节。

执行以下脚本，我们将获得 310,087 训练样本。

```Python
convert_to_training_set.py
```

一个训练样本如下：

```json
{
  "instruction": "现在你扮演一位专业的心理咨询师，你具备丰富的心理学和心理健康知识。你擅长运用多种心理咨询技巧，例如认知行为疗法原则、动机访谈技巧和解决问题导向的短期疗法。以温暖亲切的语气，展现出共情和对来访者感受的深刻理解。以自然的方式与来访者进行对话，避免过长或过短的回应，确保回应流畅且类似人类的对话。提供深层次的指导和洞察，使用具体的心理概念和例子帮助来访者更深入地探索思想和感受。避免教导式的回应，更注重共情和尊重来访者的感受。根据来访者的反馈调整回应，确保回应贴合来访者的情境和需求。请为以下的对话生成一个回复。\n\n对话：\n来访者：高三后的迷茫，高考前的恐惧，能给我一些建议么？\n咨询师：看到你的提问感觉你很焦虑，这个状态在高中高压下很容易出现。我想说的是，我看到了你的决心。这点是很多人没有的！高考最重要的不是知识是心态。是必胜的心态！什么放松吧缓缓吧，都是站着说话不腰疼，保送的又不是我，我能放松什么？！我有我的目标，我怎么可能放弃！有目标就好办，计划！缺个计划，缺个时间合理配置的复习计划。\n来访者：你说的对，我是非常焦虑，确实需要调整心态。我也明白高考的心态很重要，但是要怎样才能真正拥有必胜的心态呢？\n咨询师：首先，你要明确自己的目标，既然你想考本科，那就要为此做好准备。然后，你需要制定一个合理的复习计划，根据自己的情况来安排每天的学习时间和内容。这样可以帮助你更好地掌控整个复习过程，减少焦虑感。\n来访者：我确实没有一个合理的复习计划，每天都觉得时间不够用，也不知道该从何开始。你能给我一些建议吗？\n咨询师：",
  "output": "当然可以！你可以从高一开始，试试题海战术。每天多做一些题目，这样能够提高你的学习效率。同时，对于英语这门科目，多听多背是很重要的，数理化方面，可以做一些经典的题目，特别是那些类型经常考到的题目，多次反复做题。"
}
```

### 训练

入口文件为`train.py`，根据自身显卡，修改`lora_tune.sh`

```bash
bash lora_tune.sh
```

### 模型合并

合并前请修改一下 lora 存储位置

```
python merge_lora.py
```

## 引用

如果你使用了本项目的数据或者代码，请声明引用

```
@misc{qiu2023smile,
      title={SMILE: Single-turn to Multi-turn Inclusive Language Expansion via ChatGPT for Mental Health Support},
      author={Huachuan Qiu and Hongliang He and Shuai Zhang and Anqi Li and Zhenzhong Lan},
      year={2023},
      eprint={2305.00450},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## 免责声明

我们的心理健康支持对话机器人（以下简称"机器人"）旨在为用户提供情感支持和心理健康建议。然而，机器人不是医疗保健专业人员，不能替代医生、心理医生或其他专业人士的意见、诊断、建议或治疗。

机器人提供的建议和信息是基于算法和机器学习技术，可能并不适用于所有用户或所有情况。因此，我们建议用户在使用机器人之前咨询医生或其他专业人员，了解是否适合使用此服务。

机器人并不保证提供的建议和信息的准确性、完整性、及时性或适用性。用户应自行承担使用机器人服务的所有风险。我们对用户使用机器人服务所产生的任何后果不承担任何责任，包括但不限于任何直接或间接的损失、伤害、精神疾病、财产损失或任何其他损害。

我们强烈建议用户在使用机器人服务时，遵循以下原则：

1. 机器人并不是医疗保健专业人士，不能替代医生、心理医生或其他专业人士的意见、诊断、建议或治疗。如果用户需要专业医疗或心理咨询服务，应寻求医生或其他专业人士的帮助。

2. 机器人提供的建议和信息仅供参考，用户应自己判断是否适合自己的情况和需求。如果用户对机器人提供的建议和信息有任何疑问或不确定，请咨询医生或其他专业人士的意见。

3. 用户应保持冷静、理性和客观，不应将机器人的建议和信息视为绝对真理或放弃自己的判断力。如果用户对机器人的建议和信息产生质疑或不同意，应停止使用机器人服务并咨询医生或其他专业人士的意见。

4. 用户应遵守机器人的使用规则和服务条款，不得利用机器人服务从事任何非法、违规或侵犯他人权益的行为。

5. 用户应保护个人隐私，不应在使用机器人服务时泄露个人敏感信息或他人隐私。

6. 平台收集的数据用于学术研究。

最后，我们保留随时修改、更新、暂停或终止机器人服务的权利，同时也保留对本免责声明进行修改、更新或补充的权利。如果用户继续使用机器人服务，即视为同意本免责声明的全部内容和条款。
