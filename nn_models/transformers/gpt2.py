# nn_models/transformers/gpt2_medium.py

import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

class GPT2Medium(nn.Module):
    def __init__(self):
        super(GPT2Medium, self).__init__()
        self.config = GPT2Config.from_pretrained("gpt2-medium")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # 필수
        self.model = GPT2LMHeadModel.from_pretrained("gpt2-medium", config=self.config)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits  # <- 이게 중요함

    def generate(self, input_ids, **kwargs):
        return self.model.generate(
            input_ids=input_ids,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )

    def encode(self, text, **kwargs):
        return self.tokenizer(text, return_tensors='pt', **kwargs)

    def decode(self, token_ids, **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)

    def get_tokenizer(self):
        return self.tokenizer
