
################## INFERENCE ON MAMI DATASET ############
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=0 python3 evaluation/inference.py \
--split test \
--run_name xlm-roberta-large_batch140_10eps_1gcn \
--model_name_or_path FacebookAI/xlm-roberta-large \
--num_gcn_layers 1 \
--output_dir output_mami \
--dataset_name paoloitaliani/mami \
--input_column text \
--target_column label \
--save_affinity 


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
# --input_column text_en \
# --target_column hard_label_task4 
