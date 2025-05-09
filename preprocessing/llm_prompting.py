import os
import sys
import glob
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

def image_captioning(
    meme_path: str,
    prompt: str,
) -> str:

    # -- input prompt definition
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": meme_path,
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

def process_dataset():
    # -- loading meme dataset
    if args.huggingface_dataset:
        dataset = load_dataset(args.meme_dir)

        meme_ids = []
        meme_paths = []
        for split in dataset.keys():
            meme_ids += dataset[split]['file_name']
            meme_paths += dataset[split]['image_path']

    else:
        meme_paths = sorted( glob.glob(os.path.join(args.meme_dir, '*')) )
        meme_ids = [ os.path.basename(meme_path).split('.')[0 ]for meme_path in meme_paths]


    data = []
    issues = []
    meme_data = zip(meme_ids, meme_paths)
    for idx, (meme_id, meme_path) in enumerate(tqdm(meme_data, total=len(meme_ids))):

        try:
            prompt_a = image_captioning(meme_path, prompts['A'])
            prompt_b = image_captioning(meme_path, prompts['B'])
        except torch.cuda.OutOfMemoryError:
            prompt_a = '@MEMORY_ISSUE'
            prompt_b = '@MEMORY_ISSUE'
            issues.append( (idx, meme_path) )
            print(f'Memory issues with: {meme_path}')

        sample = (meme_id, prompt_a, prompt_b)
        data.append( sample )

        if idx == 0:
            data_df = pd.DataFrame([sample], columns=['meme_id', 'promptA', 'promptB'])
            data_df.to_csv(args.output_path, index=False)
        else:
            data_df = pd.DataFrame(data, columns=['meme_id', 'promptA', 'promptB'])
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

    issues = []
    for idx, meme_path in tqdm(issues_info):
        try:
            prompt_a = image_captioning(meme_path, prompts['A'])
            prompt_b = image_captioning(meme_path, prompts['B'])
            output_df.iloc[idx, output_df.columns.get_loc('promptA')] = prompt_a
            output_df.iloc[idx, output_df.columns.get_loc('promptB')] = prompt_b
        except torch.cuda.OutOfMemoryError:
            issues.append( (idx, meme_path) )
            print(f'Memory issues with: {meme_path}')

    output_df.to_csv(args.output_path, index=False)
    with open(args.output_path.replace('.csv', '.pkl'), 'wb') as handle:
        pickle.dump(issues, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prompt-based LLM Interaction",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--meme-dir", required=True, type=str, help="Path to where the directory containing meme images is")
    parser.add_argument('--huggingface-dataset', type=lambda x: bool(strtobool(x)), default=False, help='In case the dataset comes from HuggingFace')
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct", type=str, help="HuggingFace Model ID")
    parser.add_argument('--quant', type=lambda x: bool(strtobool(x)), default=False, help='In case model should be quantized')
    parser.add_argument('--fix-issues', type=lambda x: bool(strtobool(x)), default=False, help='In case you need to fix issues in already processed dataset')
    parser.add_argument("--output-path", required=True, type=str, help="Path to save a .csv file with the LLM responses per meme")

    args = parser.parse_args()

    # -- prompt set
    prompts = {
        'A': 'Describe this image without including what text reads and credit sources.',
        # 'B': 'You are a helpful assistant designed to detect sexist expressions or behaviours in a meme, i.e., it is sexist itself, describes a sexist situation or criticizes a sexist behaviour. Infer the implicit semantic information of the meme, considering that it may or may not contain sexist content. Please be concise (no more than three sentences) while including all relevant information.',
        'B': 'You are a helpful assistant designed to detect sexist expressions or behaviours in a meme, i.e., it is sexist itself, describes a sexist situation or criticizes a sexist behaviour. Infer the implicit semantic information of the meme, considering that it may or may not contain sexist content.',
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
