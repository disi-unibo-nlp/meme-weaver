######################## TRAINING ON MEMES_EXIST 2024 DATASET ############
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=1 python3 src/train_hf.py \
# --num_gcn_layers 1 \
# --no_peft \
# --add_caption \
# --run_name xlm-roberta-large_texten_qwen25vl_caption_140batch_10eps_mami140_1gcn \
# --input_column text_en \
# --target_column hard_label_task4 \
# --logging online \
# --do_train \
# --do_eval \
# --do_predict \
# --output_dir output_hard_label_task4 \
# --model_name_or_path output_mami/xlm-roberta-large_batch140_10eps_1gcn/checkpoint-560  \
# --dataset_subset default \
# --dataset_name paoloitaliani/memes_exist2024 \
# --log_level error \
# --gradient_accumulation_steps 1 \
# --max_seq_length 514 \
# --learning_rate 5e-6 \
# --num_train_epochs 10 \
# --save_strategy steps \
# --evaluation_strategy steps \
# --fp16 \
# --gradient_checkpointing \
# --load_best_model_at_end \
# --overwrite_cache \
# --save_total_limit 1 \
# --weight_decay 0.01 \
# --label_smoothing_factor 0.1 \
# --remove_unused_columns \
# --metric_for_best_model accuracy \
# --per_device_train_batch_size 140 \
# --per_device_eval_batch_size 140 


# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=1 python3 src/train_hf.py \
# --no_peft \
# --add_caption \
# --run_name xlm-roberta-large_texten_qwen25vl_caption_140batch_10eps_mami140 \
# --input_column text_en \
# --target_column hard_label_task4 \
# --logging online \
# --do_train \
# --do_eval \
# --do_predict \
# --output_dir output_hard_label_task4 \
# --model_name_or_path output_mami/xlm-roberta-large_batch140_10eps_1gcn/checkpoint-560  \
# --dataset_subset default \
# --dataset_name paoloitaliani/memes_exist2024 \
# --log_level error \
# --gradient_accumulation_steps 1 \
# --max_seq_length 514 \
# --learning_rate 5e-6 \
# --num_train_epochs 10 \
# --save_strategy steps \
# --evaluation_strategy steps \
# --fp16 \
# --gradient_checkpointing \
# --load_best_model_at_end \
# --overwrite_cache \
# --save_total_limit 1 \
# --weight_decay 0.01 \
# --label_smoothing_factor 0.1 \
# --remove_unused_columns \
# --metric_for_best_model accuracy \
# --per_device_train_batch_size 140 \
# --per_device_eval_batch_size 140 

######################## TRAINING ON MAMI DATASET ############
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=1 python3 src/train_hf.py \
--num_gcn_layers 2 \
--custom_gcn learn_upd \
--no_peft \
--run_name  xlm-roberta-large_batch140_10eps_2gcn_1.2upd \
--input_column text \
--target_column label \
--logging online \
--do_train \
--do_eval \
--do_predict \
--output_dir output_mami \
--model_name_or_path  FacebookAI/xlm-roberta-large \
--dataset_name paoloitaliani/mami \
--log_level error \
--gradient_accumulation_steps 1 \
--max_seq_length 514 \
--learning_rate 5e-6 \
--num_train_epochs 10 \
--save_strategy steps \
--evaluation_strategy steps \
--fp16 \
--gradient_checkpointing \
--load_best_model_at_end \
--overwrite_cache \
--save_total_limit 1 \
--weight_decay 0.01 \
--label_smoothing_factor 0.1 \
--remove_unused_columns \
--metric_for_best_model accuracy \
--per_device_train_batch_size 140 \
--per_device_eval_batch_size 140
