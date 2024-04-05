from glm2.modeling_chatglm import ChatGLMForConditionalGeneration as ChatGLM2ForConditionalGeneration
from glm2.tokenization_chatglm import ChatGLMTokenizer as ChatGLM2Tokenizer
from glm2.configuration_chatglm import ChatGLMConfig as ChatGLM2Config

from utils import GLM2PromptDataSet

MODE = {"glm2": {"model": ChatGLM2ForConditionalGeneration, "tokenizer": ChatGLM2Tokenizer, "config": ChatGLM2Config,
                 "dataset": GLM2PromptDataSet}}
