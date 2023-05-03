<div align="center">

<h2>
    SMILE: Single-turn to Multi-turn Inclusive Language Expansion via ChatGPT for Mental Health Support
</h2>

<a href='https://arxiv.org/pdf/2305.00450.pdf'><img src='https://img.shields.io/badge/ArXiv-2305.00450-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='https://qiuhuachuan.github.io/smile'><img src='https://img.shields.io/badge/Project-Page-Green'></a>

<div>
    <a href='https://scholar.google.com/citations?user=UCx7h5YAAAAJ&hl=en' target='_blank'>Huachuan Qiu <sup>1, 2</sup></a>&emsp;
    <span>Hongliang He <sup>1, 2</sup></span>&emsp;
    <span>Shuai Zhang <sup>1, 2</sup></span>&emsp;
    <span>Anqi Li <sup>1, 2</sup></span>&emsp;
    <a href='https://scholar.google.com/citations?user=tlDABkgAAAAJ&hl=en&oi=ao' target='_blank'>Zhenzhong Lan <sup>2</sup></a>&emsp;
</div>

<br>

<div>
    <sup>1</sup> Zhejiang University &emsp; <sup>2</sup> Westlake University &emsp;
</div>

<br>
<br>

<b>TL;DR: An efficient method that can:</b>  
1️⃣ automatically construct a collection of large-scale, diverse and muti-turn conversations;<br>
2️⃣ provide an SmileChat datatset for building a dialog agent for mental health support.

<br>

</div>

# Abstract

> There has been an increasing research interest in developing specialized dialogue systems that can offer mental health support. However, gathering large-scale and real-life multi-turn conversations for mental health support poses challenges due to the sensitivity of personal information, as well as the time and cost involved. To address these issues, we introduce the SMILE approach, an inclusive language expansion technique that employs ChatGPT to extend public single-turn dialogues into multi-turn ones. Our research first presents a preliminary exploratory study that validates the effectiveness of the SMILE approach. Furthermore, we conduct a comprehensive and systematic contrastive analysis of datasets generated with and without the SMILE approach, demonstrating that the SMILE method results in a large-scale, diverse, and close-to-real-life multi-turn mental health support conversation corpus, including dialog topics, lexical and semantic features. Finally, we use the collected corpus (SMILECHAT) to develop a more effective dialogue system that offers emotional support and constructive suggestions in multi-turn conversations for mental health support.

# File Structure

- The data directory contains three folders: plain, smile, and smile_cot.
- All dialogues can be found in the data directory.
- An example selected using the smile method is shown below. For the English translation, please refer to our paper.

```JSON
[
  "求助者：最近总是和妈妈闹矛盾，但是又不知道该怎么办，能帮我想想办法吗？",
  "支持者：我了解你的情况，跟亲人之间经常会产生矛盾是很常见的现象。你不妨试试和妈妈沟通一下，平静地提出自己的疑惑和不满，看看能否解决矛盾。",
  "求助者：但是每次我和妈妈说话，总会起争端，她总是让我感觉她不信任我，我该怎么办呢？",
  "支持者：听起来你和妈妈之间的交流很困难，你可以试试换个方式和她沟通，比如写信或者找一个更加中立的人一起协调谈话，让大家都有更好的表达机会。",
  "求助者：我特别讨厌和她吵架，可是我有时候就是自制力不够，很难抑制自己的情绪。",
  "支持者：青春期的年轻人情绪波动很大很正常，但是你可以试试找些方法来缓解情绪，比如听听音乐、看看书等等，使自己情绪更稳定。",
  "求助者：妈妈总是很为我担心，但是我感觉她的表达方式让我很不舒服，怎么办？",
  "支持者：你可以试着跟妈妈提出你的感受，说出你觉得她的表达方式不太适合你，看看一起可以找到一个更好的沟通方式。",
  "求助者：近期我迷上了游戏，可是妈妈总是担心我的学业，会经常跟我谈中考和未来，我也很焦虑。",
  "支持者：我能理解你的压力，但是你的妈妈对你的学业担忧也是很正常的。你可以试着和妈妈沟通一下，表明自己的压力和困惑，寻求她的理解和支持。",
  "求助者：妈妈总是说我顶嘴顶的不好，可是我并没有说过什么不好的话，这些误解让我很难受。",
  "支持者：很抱歉听到这些误解带给你的困扰，你可以试着和妈妈沟通，表明你没有说过不好的话，避免误解的发生。",
  "求助者：有时候我觉得妈妈很不公平，总是让我做家务和学习，而她却不怎么做，这让我很不满意。",
  "支持者：家务和学习的确是每个人都需要承担的责任，但是你可以跟妈妈商量一下，建立更合理的分工方式，让大家的负担更加均衡。",
  "求助者：我有时候会想，如果我不在妈妈身边，她就不会那么疲惫与辛苦了，是不是我应该离开她，这样她就会开心一些。",
  "支持者：不要把这些想法一直压在心里，试着跟她好好沟通，说说你的担心和顾虑，寻求她的支持和理解。离开并不会解决问题，关心和照顾妈妈也是你的责任之一。",
  "求助者：非常感谢你的耐心倾听和建议，我会好好尝试和妈妈沟通，解决我们之间的矛盾。",
  "支持者：很高兴能帮助你，你已经迈出了寻求帮助的第一步，接下来只要继续坚持下去，相信问题一定会得到好的解决。"
]
```

# Citation

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

# For Training

## Step 1: Convert to Training File

We will obtain 355733 training examples.

```Python
python convert_to_train_json.py
```

## DONE

- [x] Release SmileChat dataset

## ⏳ TODO

- [ ] Release analysis code
- [ ] Release training code

<br>
