import os
import json
import base64
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from datasets import load_dataset
from sklearn.metrics import (
    f1_score,
    accuracy_score,
)

OPENAI_API_KEY = "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"


def get_sys_prmpt_exist_wtext(ocr_text, caption_text):
    sys_instruction_classification = f"""
                                        Consider the following definitions:
                                        1. ‘SEXIST’ - “The meme is sexist itself, describes a sexist situation or criticizes a sexist behavior. ”
                                        2. ‘NON-SEXIST’ - “The meme does not prejudice, underestimate, or discriminate women.”
                                        Consider also the following extracted text from the image and prompt-based image caption:
                                        Extracted text from the image: {ocr_text}.
                                        Prompt-based image caption: {caption_text}.
                                        Classify the input meme as ‘SEXIST’ or ‘NON-SEXIST’ based on the above definitions considering the image as well as the extracted text from the image. Provide the answer as either ‘SEXIST’ or ‘NON-SEXIST’ only.
                                        """

    return sys_instruction_classification


def get_sys_prmpt_exist():
    sys_instruction_classification = f"""
                                        Consider the following definitions:
                                        1. ‘SEXIST’ - “The meme is sexist itself, describes a sexist situation or criticizes a sexist behavior.”
                                        2. ‘NON-SEXIST’ - “The meme does not prejudice, underestimate, or discriminate women.”
                                        Classify the input meme as ‘SEXIST’ or ‘NON-SEXIST’ based on the above definitions and considering the image. Provide the answer as either ‘SEXIST’ or ‘NON-SEXIST’ only.
                                        """

    return sys_instruction_classification


def get_sys_prmpt_mami_wtext(ocr_text, caption_text):
    sys_instruction_classification = f"""
                                        Consider the following definitions. 1. 'MISOGYNISTIC' - 'The meme expresses hatred, hostility, or contempt toward women, including insulting, demeaning, or promoting harm or discrimination against them.' 2. 'NON-MISOGYNISTIC' - 'The meme does not express hatred, hostility, or contempt, and does not promote harm, discrimination, or demeaning attitudes against women.' Classify the input meme as 'MISOGYNISTIC' or 'NON-MISOGYNISTIC' based on the above definitions, considering the image as well as the extracted text from the image. Provide the answer as either 'MISOGYNISTIC' or 'NON-MISOGYNISTIC' only. Extracted text from the image: {ocr_text}.
                                        """

    return sys_instruction_classification


def get_sys_prmpt_mami():
    sys_instruction_classification = f"""Consider the following definitions. 1. 'MISOGYNISTIC' - 'The meme expresses hatred, hostility, or contempt toward women, including insulting, demeaning, or promoting harm or discrimination against them.' 2. 'NON-MISOGYNISTIC' - 'The meme does not express hatred, hostility, or contempt, and does not promote harm, discrimination, or demeaning attitudes against women.' Classify the input meme as 'MISOGYNISTIC' or 'NON-MISOGYNISTIC' based on the above definitions considering the image. Provide the answer as either 'MISOGYNISTIC' or 'NON-MISOGYNISTIC' only."""

    return sys_instruction_classification

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def make_completion_openai(system_instruction, base64_image, client):
    response = client.responses.create(
        model=args.model,
        input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": system_instruction.replace("\n", " ").strip()},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                }
            ],
        )

    completion = response.output_text

    return completion


def build_system_prompt(item):
    """Return the correct system prompt based on dataset and whether text is used."""
    is_exist = "exist" in args.dataset_name.lower()

    if args.use_text:
        text = item[args.text_column]
        caption = item["caption_promptA"]
        prompt_fn = get_sys_prmpt_exist_wtext if is_exist else get_sys_prmpt_mami_wtext
        return prompt_fn(text, caption)

    prompt_fn = get_sys_prmpt_exist if is_exist else get_sys_prmpt_mami
    return prompt_fn()


def load_image_base64(item):
    """Locate the image file and return it encoded as base64."""
    is_exist = "exist" in args.dataset_name.lower()

    if is_exist:
        meme_id = item["id"]
        base_dir = Path("exist2024_memes_dataset")
        for split in ("test", "training"):
            candidate = base_dir / split / "memes" / meme_id
            if candidate.exists():
                return encode_image(candidate)
        raise FileNotFoundError(f"Image for id '{meme_id}' not found in test/ or training/ folders.")
    else:
        path = Path("MAMI_Dataset") / "MAMI_2022_images" / item["file_name"]
        return encode_image(path)


def run_on_dataset(dataset, completer, client):
    results = []
    for item in tqdm(dataset):
        system_prompt = build_system_prompt(item)
        base64_image = load_image_base64(item)

        completion = completer(system_prompt, base64_image, client)
        output_dict = {"label": item[args.target_column], "prediction": completion.strip()}

        results.append(output_dict)

    return results


def evaluate_results():
    use_text = "use_text" if args.use_text else "no_text"
    output_path = os.path.join(args.output_dir, f"{args.model}_{use_text}.json")
    # read predictions
    with open(output_path, "r") as f:
        results = json.load(f)

    # pull the test split (so we can get 'lang' in the same order)
    test = load_dataset(args.dataset_name)["test"]

    labels = []
    preds = []
    for result in results:
        labels.append(result["label"])
        if "exist" in args.dataset_name:
            p = 0 if "non-sexist" in result["prediction"].strip().lower() else 1
        else:
            p = 0 if "non-misogynistic" in result["prediction"].strip().lower() else 1
        preds.append(p)

    # build a DataFrame tying each prediction back to its language
    df = pd.DataFrame({
        "ground_truth": labels,
        "prediction": preds,
        "lang": [ex["lang"] for ex in test]  # assumes test examples are in the same order
    })

    for lang in ["en", "es"]:
        sub = df[df["lang"] == lang]
        f1 = f1_score(sub["ground_truth"], sub["prediction"],
                      average="macro", zero_division=0)
        acc = accuracy_score(sub["ground_truth"], sub["prediction"])
        print(
            f"{lang.capitalize():>8} — "
            f"F1-macro: {f1:.3f}, "
            f"Accuracy: {acc:.3f}, "
        )

    f1 = f1_score(df["ground_truth"], df["prediction"],
                  average="macro", zero_division=0)
    acc = accuracy_score(df["ground_truth"], df["prediction"])

    print("Overall — ",
          f"F1-macro: {f1:.3f}, "
          f"Accuracy: {acc:.3f}, ")


def main():

    client = OpenAI(
        api_key=OPENAI_API_KEY
    )

    test_set = load_dataset(args.dataset_name)["test"]
    results = run_on_dataset(test_set, make_completion_openai, client)

    use_text = "use_text" if args.use_text else "no_text"

    output_path = os.path.join(args.output_dir, f"{args.model}_{use_text}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="LLM Evaluation")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--text_column", type=str)
    parser.add_argument("--target_column", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--use_text", action="store_true")

    args = parser.parse_args()

    # main()
    evaluate_results()