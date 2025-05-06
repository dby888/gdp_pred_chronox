from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "amazon/chronos-t5-large"
local_dir = "C:/Users/dby/huggingface_models/chronos-t5-large"

# 下载模型
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=local_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir)