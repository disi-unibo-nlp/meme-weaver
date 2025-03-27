import os
import torch
import argparse
import numpy as np
from  tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel



def main():
    model_name = args.embed_model.split("/")[-1].replace("-", "_")
    output_folder = os.path.join("data", "embeddings", model_name)
    os.makedirs(output_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the dataset
    dataset = load_dataset('paoloitaliani/memes_exist2024')
    model = AutoModel.from_pretrained(args.embed_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.embed_model)

    # Set model to evaluation mode.
    model.eval()

    # Process each dataset split.
    for split in ["train", "validation", "test"]:
        embeddings = []
        ids = []
        texts = dataset[split][args.feature]
        id_exist = dataset[split]['id_EXIST']  # Assumes the column name is 'id_EXIST'
        
        for text, id_val in tqdm(zip(texts, id_exist), total=len(texts)):
            # Tokenize the text with truncation and padding.
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            
            # Forward pass without computing gradients.
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Extract the embedding from the first token ([CLS]) and move to CPU.
            embedding = outputs.last_hidden_state[:, 0, :].cpu().detach().numpy()
            embeddings.append(embedding)
            ids.append(id_val)
        
        # Save both the ids and embeddings using NumPy's savez to store them together.
        
        output_path = os.path.join(output_folder, f"{split}_embeddings.npz")
        np.savez(output_path, ids=np.array(ids), embeddings=np.array(embeddings))
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_model", default="FacebookAI/xlm-roberta-large")
    parser.add_argument("--feature", default="text")

    args = parser.parse_args()
    main()