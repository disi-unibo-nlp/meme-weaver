# ######################## TRAINING ON MAMI DATASET ############

# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=2 python3 src/train_multimodal_classifier.py \
# --num_gcn_layers 1 \
# --no_peft \
# --run_name clip-vit-large_batch20_5e-6lr_1gcn_xuinit_concat_test2 \
# --logging disabled \
# --target_column label \
# --text_column text \
# --image_column image_path \
# --id_column file_name \
# --do_train \
# --do_eval \
# --do_predict \
# --output_dir output_mami \
# --model_name_or_path openai/clip-vit-large-patch14 \
# --dataset_name paoloitaliani/mami \
# --log_level error \
# --gradient_accumulation_steps 1 \
# --max_seq_length 514 \
# --learning_rate 5e-6 \
# --num_train_epochs 10 \
# --save_strategy steps \
# --eval_strategy steps \
# --fp16 \
# --gradient_checkpointing \
# --load_best_model_at_end \
# --overwrite_cache \
# --save_total_limit 1 \
# --weight_decay 0.01 \
# --label_smoothing_factor 0.1 \
# --remove_unused_columns \
# --metric_for_best_model accuracy \
# --per_device_train_batch_size 20 \
# --per_device_eval_batch_size 20



# ######################## TRAINING ON MEMES_EXIST 2024 DATASET ############
CUDA_LAUNCH_BLOCKING=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=2 python3 src/train_qwen_classifier.py \
--image_caption caption_promptA \
--no_peft \
--run_name Qwen2.5-1.5B_test \
--logging disabled \
--target_column hard_label_task4 \
--text_column text_en \
--image_column image \
--id_column id \
--do_train \
--do_eval \
--do_predict \
--output_dir output_hard_label_task4 \
--model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
--dataset_name paoloitaliani/memes_exist2024 \
--dataset_subset default \
--log_level error \
--gradient_accumulation_steps 1 \
--max_seq_length 514 \
--learning_rate 5e-6 \
--num_train_epochs 10 \
--save_strategy steps \
--eval_strategy steps \
--fp16 \
--gradient_checkpointing \
--load_best_model_at_end \
--overwrite_cache \
--save_total_limit 1 \
--weight_decay 0.01 \
--remove_unused_columns \
--metric_for_best_model loss \
--per_device_train_batch_size 5 \
--per_device_eval_batch_size 5 \
--max_train_samples 100 \
--max_eval_samples 10 \
--max_predict_samples 10 \


######################## TRAINING ON MMHS150K DATASET ############

# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=3 python3 src/train_multimodal_classifier.py \
# --num_gcn_layers 1 \
# --no_peft \
# --run_name clip-vit-large_batch120_5e-6lr_1gcn_xuinit_concat_3eps \
# --logging online \
# --target_column label \
# --text_column text \
# --image_column image \
# --id_column tweet_id \
# --do_train \
# --do_eval \
# --do_predict \
# --output_dir output_mmhs150k \
# --model_name_or_path openai/clip-vit-large-patch14 \
# --dataset_name paoloitaliani/mmhs150k \
# --log_level error \
# --gradient_accumulation_steps 1 \
# --max_seq_length 514 \
# --learning_rate 5e-6 \
# --num_train_epochs 3 \
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
# --per_device_train_batch_size 120 \
# --per_device_eval_batch_size 120 
# # --max_train_samples 100 \
# # --max_eval_samples 10 \
# # --max_predict_samples 10

# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=3 python3 src/train_multimodal_classifier.py \
# --num_gcn_layers 1 \
# --no_peft \
# --run_name clip-vit-large_batch150_5e-6lr_1gcn_xuinit_concat_3eps \
# --logging online \
# --target_column label \
# --text_column text \
# --image_column image \
# --id_column tweet_id \
# --do_train \
# --do_eval \
# --do_predict \
# --output_dir output_mmhs150k \
# --model_name_or_path openai/clip-vit-large-patch14 \
# --dataset_name paoloitaliani/mmhs150k \
# --log_level error \
# --gradient_accumulation_steps 1 \
# --max_seq_length 514 \
# --learning_rate 5e-6 \
# --num_train_epochs 3 \
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
# --per_device_train_batch_size 150 \
# --per_device_eval_batch_size 150 
# # --max_train_samples 100 \
# # --max_eval_samples 10 \
# # --max_predict_samples 10

# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=3 python3 src/train_multimodal_classifier.py \
# --num_gcn_layers 1 \
# --no_peft \
# --run_name clip-vit-large_batch180_5e-6lr_1gcn_xuinit_concat_3eps \
# --logging online \
# --target_column label \
# --text_column text \
# --image_column image \
# --id_column tweet_id \
# --do_train \
# --do_eval \
# --do_predict \
# --output_dir output_mmhs150k \
# --model_name_or_path openai/clip-vit-large-patch14 \
# --dataset_name paoloitaliani/mmhs150k \
# --log_level error \
# --gradient_accumulation_steps 1 \
# --max_seq_length 514 \
# --learning_rate 5e-6 \
# --num_train_epochs 3 \
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
# --per_device_train_batch_size 180 \
# --per_device_eval_batch_size 180 
# # --max_train_samples 100 \
# # --max_eval_samples 10 \
# # --max_predict_samples 10

# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=3 python3 src/train_multimodal_classifier.py \
# --num_gcn_layers 1 \
# --no_peft \
# --run_name clip-vit-large_batch200_5e-6lr_1gcn_xuinit_concat_3eps \
# --logging online \
# --target_column label \
# --text_column text \
# --image_column image \
# --id_column tweet_id \
# --do_train \
# --do_eval \
# --do_predict \
# --output_dir output_mmhs150k \
# --model_name_or_path openai/clip-vit-large-patch14 \
# --dataset_name paoloitaliani/mmhs150k \
# --log_level error \
# --gradient_accumulation_steps 1 \
# --max_seq_length 514 \
# --learning_rate 5e-6 \
# --num_train_epochs 3 \
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
# --per_device_train_batch_size 200 \
# --per_device_eval_batch_size 200 
# # --max_train_samples 100 \
# # --max_eval_samples 10 \
# # --max_predict_samples 10


# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=3 python3 src/train_multimodal_classifier.py \
# --num_gcn_layers 1 \
# --no_peft \
# --run_name clip-vit-large_batch230_5e-6lr_1gcn_xuinit_concat_3eps \
# --logging online \
# --target_column label \
# --text_column text \
# --image_column image \
# --id_column tweet_id \
# --do_train \
# --do_eval \
# --do_predict \
# --output_dir output_mmhs150k \
# --model_name_or_path openai/clip-vit-large-patch14 \
# --dataset_name paoloitaliani/mmhs150k \
# --log_level error \
# --gradient_accumulation_steps 1 \
# --max_seq_length 514 \
# --learning_rate 5e-6 \
# --num_train_epochs 3 \
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
# --per_device_train_batch_size 230 \
# --per_device_eval_batch_size 230 
# # --max_train_samples 100 \
# # --max_eval_samples 10 \
# # --max_predict_samples 10

# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=3 python3 src/train_multimodal_classifier.py \
# --num_gcn_layers 1 \
# --no_peft \
# --run_name clip-vit-large_batch260_5e-6lr_1gcn_xuinit_concat_3eps \
# --logging online \
# --target_column label \
# --text_column text \
# --image_column image \
# --id_column tweet_id \
# --do_train \
# --do_eval \
# --do_predict \
# --output_dir output_mmhs150k \
# --model_name_or_path openai/clip-vit-large-patch14 \
# --dataset_name paoloitaliani/mmhs150k \
# --log_level error \
# --gradient_accumulation_steps 1 \
# --max_seq_length 514 \
# --learning_rate 5e-6 \
# --num_train_epochs 3 \
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
# --per_device_train_batch_size 260 \
# --per_device_eval_batch_size 260 
# # --max_train_samples 100 \
# # --max_eval_samples 10 \
# # --max_predict_samples 10