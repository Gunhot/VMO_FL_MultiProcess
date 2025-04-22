from nn_models.transformers.gpt2 import GPT2Medium
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch

# 모델 로드
config = GPT2Config.from_pretrained("gpt2-medium")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token  # 필수
model = GPT2LMHeadModel.from_pretrained("gpt2-medium", config=config)

model_path = "checkpoints/server_model_round_5.pt"
model = GPT2Medium()
model.load_state_dict(torch.load(model_path))
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 입력 문장
input_text = "What is the capital of France?"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 반복 생성 파라미터
max_new_tokens = 50  # 생성할 최대 토큰 수
generated = input_ids

with torch.no_grad():
    for _ in range(max_new_tokens):
        outputs = model(generated)
        logits = outputs[1] if isinstance(outputs, tuple) else outputs.logits
        next_token_logits = logits[:, -1, :]  # 마지막 토큰에 대한 로짓
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

        generated = torch.cat((generated, next_token_id), dim=1)

# 디코딩해서 출력
output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
print("Generated text:\n", output_text)
