from datasets import load_dataset
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token

print("🔹 Loading VIGGO dataset...")
dataset = load_dataset("GEM/viggo", split="train", trust_remote_code=True)

print("✅ Loaded", len(dataset), "samples")

sample = dataset[0]
print("📘 Sample keys:", sample.keys())
print("🔸 Meaning Representation:", sample["meaning_representation"])
print("🔸 Reference (Target):", sample.get("human_reference", sample.get("reference", "MISSING")))

input_text = "MR: " + sample["meaning_representation"]
target_text = sample.get("human_reference", sample.get("reference", ""))

input_ids = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)["input_ids"]
label_ids = tokenizer(target_text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)["input_ids"]

print("🧾 Tokenized input:", input_ids.shape)
print("🧾 Tokenized label:", label_ids.shape)
print("🔓 Decoded input:", tokenizer.decode(input_ids[0]))
print("🔓 Decoded label:", tokenizer.decode(label_ids[0]))
