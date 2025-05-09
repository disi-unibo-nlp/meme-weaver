import os
import sys
import glob
import json
import torch
import pickle
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from distutils.util import strtobool
from transformers import BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

def multimodal_tweet_captioning(
    tweet_text: str,
    img_path: str,
    prompt: str,
) -> str:

    # -- input prompt definition
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": tweet_text},
                {
                    "type": "image",
                    "image": img_path,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # -- forward pass
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    assert len(output_text) == 1, 'More than one output?'

    return output_text[0].strip().replace('\n', ' ').replace('\r', '')

def text_tweet_captioning(
    tweet_text: str,
    prompt: str,
) -> str:

    # -- input prompt definition
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": tweet_text},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # -- forward pass
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    assert len(output_text) == 1, 'More than one output?'

    return output_text[0].strip().replace('\n', ' ').replace('\r', '')

def process_dataset():
    # -- loading img dataset
    if args.huggingface_dataset:
        dataset = load_dataset(args.img_dir)

        img_ids = []
        img_paths = []
        for split in dataset.keys():
            img_ids += dataset[split]['file_name']
            img_paths += dataset[split]['image_path']

    else:
        img_paths = sorted( glob.glob(os.path.join(args.img_dir, '*')) )
        img_ids = [ os.path.basename(img_path).split('.')[0] for img_path in img_paths]
        with open('./MMHS150K/MMHS150K_GT.json', 'r') as f:
            metadata = json.load(f)

    data = []
    issues = []
    img_data = zip(img_ids, img_paths)
    for idx, (img_id, img_path) in enumerate(tqdm(img_data, total=len(img_ids))):
        tweet_text = metadata[img_id]["tweet_text"]
        try:
            prompt_c = multimodal_tweet_captioning(tweet_text, img_path, prompts['C'])
            prompt_d = text_tweet_captioning(tweet_text, prompts['D'])
        except torch.cuda.OutOfMemoryError:
            prompt_c = '@MEMORY_ISSUE'
            prompt_d = '@MEMORY_ISSUE'
            issues.append( (idx, img_path) )
            print(f'Memory issues with: {img_path}')

        sample = (img_id, prompt_c, prompt_d)
        data.append( sample )

        if idx == 0:
            data_df = pd.DataFrame([sample], columns=['img_id', 'promptC', 'promptD'])
            data_df.to_csv(args.output_path, index=False)
        else:
            data_df = pd.DataFrame(data, columns=['img_id', 'promptC', 'promptD'])
            data_df.loc[[idx]].to_csv(
                args.output_path,
                index=False,
                header=False,
                mode='a',
            )

    with open(args.output_path.replace('.csv', '.pkl'), 'wb') as handle:
        pickle.dump(issues, handle, protocol=pickle.HIGHEST_PROTOCOL)

def fix_issues():
    output_df = pd.read_csv(args.output_path)
    with open(args.output_path.replace('.csv', '.pkl'), 'rb') as handle:
        issues_info = pickle.load(handle)
    with open('./MMHS150K/MMHS150K_GT.json', 'r') as f:
        metadata = json.load(f)

    issues = []
    for idx, img_path in tqdm(issues_info):
        tweet_text = metadata[os.path.basename(img_path).split('.')[0]]["tweet_text"]
        try:
            prompt_c = multimodal_tweet_captioning(tweet_text, img_path, prompts['C'])
            prompt_d = text_tweet_captioning(tweet_text, prompts['D'])
            output_df.iloc[idx, output_df.columns.get_loc('promptC')] = prompt_c
            output_df.iloc[idx, output_df.columns.get_loc('promptD')] = prompt_d
        except torch.cuda.OutOfMemoryError:
            issues.append( (idx, img_path) )
            print(f'Memory issues with: {img_path}')

    output_df.to_csv(args.output_path, index=False)
    with open(args.output_path.replace('.csv', '.pkl'), 'wb') as handle:
        pickle.dump(issues, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prompt-based LLM Interaction",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--img-dir", required=True, type=str, help="Path to where the directory containing img images is")
    parser.add_argument('--huggingface-dataset', type=lambda x: bool(strtobool(x)), default=False, help='In case the dataset comes from HuggingFace')
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct", type=str, help="HuggingFace Model ID")
    parser.add_argument('--quant', type=lambda x: bool(strtobool(x)), default=False, help='In case model should be quantized')
    parser.add_argument('--fix-issues', type=lambda x: bool(strtobool(x)), default=False, help='In case you need to fix issues in already processed dataset')
    parser.add_argument("--output-path", required=True, type=str, help="Path to save a .csv file with the LLM responses per img")

    args = parser.parse_args()

    # -- prompt set
    prompts = {
        'C': 'You are a helpful assistant designed to detect hate speech expressions or behaviours in a tweet, which consists of both text and an associated image. Infer the implicit semantic information of the multimodal tweet, considering that it may or may not contain hate speech content. If applicable, identify whether the tweet targets a specific demographic group. Please be concise (no more than three sentences) while including all relevant information',
        'D': 'You are a helpful assistant designed to detect hate speech. Infer the implicit semantic information of the following text targeted a certain demographic group. Please begin with "the text contains" in your response. Text: ',
    }

    # -- llm model building
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype="auto",
        quantization_config=quantization_config if args.quant else None,
    )

    processor = AutoProcessor.from_pretrained(args.model_id)

    ## -- dataset processing
    if args.fix_issues:
        fix_issues()
    else:
        process_dataset()
