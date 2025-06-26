
################## INFERENCE ON MAMI DATASET MULTIMODAL ############
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=1 python3 evaluation/inference_multimodal.py \
# --split test \
# --id_column file_name \
# --run_name clip-vit-large_batch20_5e-6lr \
# --model_name_or_path openai/clip-vit-large-patch14 \
# --output_dir output_mami \
# --dataset_name paoloitaliani/mami \
# --target_column label \
# --text_column text \
# --image_column image_path \
# --save_affinity \
# --per_device_eval_batch_size 40

############ INFERENCE ON MEMES_EXIST 2024 DATASET ############

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=0 python3 evaluation/inference_multimodal.py \
--split test \
--run_name  clip-vit-large_batch64_5e-6lr_1gcn_xuinit_mami20bs_captionPA_concat \
--model_name_or_path openai/clip-vit-large-patch14 \
--output_dir output_hard_label_task4 \
--dataset_name paoloitaliani/memes_exist2024 \
--target_column hard_label_task4 \
--text_column text_en \
--image_column image \
--save_affinity \
--per_device_eval_batch_size 64
