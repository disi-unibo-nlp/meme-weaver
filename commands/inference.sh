
# ################## INFERENCE ON MAMI DATASET ############
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=2 python3 evaluation/inference.py \
# --split test \
# --run_name xlm-roberta-large_batch100_10eps_reprod_1gcn_seed45 \
# --model_name_or_path FacebookAI/xlm-roberta-large \
# --output_dir output_mami \
# --dataset_name paoloitaliani/mami \
# --text_column text \
# --target_column label \
# --save_affinity \
# --per_device_eval_batch_size 100

################## INFERENCE ON MAMI DATASET MULTIMODAL ############
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=2 python3 evaluation/inference_multimodal.py \
--split test \
--run_name clip-vit-large_batch100_5e-6lr_1gcn \
--model_name_or_path openai/clip-vit-large-patch14 \
--output_dir output_mami \
--dataset_name paoloitaliani/mami \
--target_column label \
--text_column text \
--image_column image_path \
--per_device_eval_batch_size 100

############ INFERENCE ON MEMES_EXIST 2024 DATASET ############
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=0 python3 evaluation/inference.py \
# --split test \
# --run_name xlm-roberta-large_texten_qwen25vl_caption_140batch_10eps_mami140_1gcn \
# --model_name_or_path FacebookAI/xlm-roberta-large \
# --num_gcn_layers 1 \
# --add_caption \
# --output_dir output_hard_label_task4 \
# --dataset_name paoloitaliani/memes_exist2024 \
# --dataset_subset default \
# --text_column text_en \
# --target_column hard_label_task4 
