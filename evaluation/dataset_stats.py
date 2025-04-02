from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("paoloitaliani/memes_exist2024")
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")



token_lengths = []
# Loop through all splits
for split in dataset.keys():
    for example in dataset[split]:
        full_text = example["text_en"] + "[CPT]" + example["qwen25vl_caption"]
        tokens = tokenizer.encode(full_text, truncation=False)
        token_lengths.append((len(tokens)))

# Sort and get top 10
top_10 = sorted(token_lengths, key=lambda x: x, reverse=True)[:10]

print(top_10)
