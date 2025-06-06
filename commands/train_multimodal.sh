######################## TRAINING ON MAMI DATASET ############
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=2 python3 src/train_multimodal_classifier.py \
# --num_gcn_layers 1 \
# --image_caption caption_promptA \
# --no_peft \
# --run_name clip-vit-large_batch20_5e-6lr_1gcn_xuinit_captionPA \
# --logging online \
# --target_column label \
# --text_column text \
# --image_column image \
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
# --per_device_train_batch_size 20 \
# --per_device_eval_batch_size 20
# # --max_train_samples 100 \
# # --max_eval_samples 10 \
# # --max_predict_samples 10

# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=2 python3 src/train_multimodal_classifier.py \
# --num_gcn_layers 1 \
# --image_caption caption_promptB \
# --no_peft \
# --run_name clip-vit-large_batch20_5e-6lr_1gcn_xuinit_captionPB \
# --logging online \
# --target_column label \
# --text_column text \
# --image_column image \
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
# --per_device_train_batch_size 20 \
# --per_device_eval_batch_size 20
# # --max_train_samples 100 \
# # --max_eval_samples 10 \
# # --max_predict_samples 10

# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=2 python3 src/train_multimodal_classifier.py \
# --num_gcn_layers 1 \
# --no_peft \
# --run_name clip-vit-large_batch20_2e-6lr_1gcn_xuinit \
# --logging online \
# --target_column label \
# --text_column text \
# --image_column image \
# --do_train \
# --do_eval \
# --do_predict \
# --output_dir output_mami \
# --model_name_or_path openai/clip-vit-large-patch14 \
# --dataset_name paoloitaliani/mami \
# --log_level error \
# --gradient_accumulation_steps 1 \
# --max_seq_length 514 \
# --learning_rate 2e-6 \
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
# --per_device_train_batch_size 20 \
# --per_device_eval_batch_size 20
# # --max_train_samples 100 \
# # --max_eval_samples 10 \
# # --max_predict_samples 10

# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=2 python3 src/train_multimodal_classifier.py \
# --num_gcn_layers 1 \
# --no_peft \
# --run_name clip-vit-large_batch20_1e-6lr_1gcn_xuinit \
# --logging online \
# --target_column label \
# --text_column text \
# --image_column image \
# --do_train \
# --do_eval \
# --do_predict \
# --output_dir output_mami \
# --model_name_or_path openai/clip-vit-large-patch14 \
# --dataset_name paoloitaliani/mami \
# --log_level error \
# --gradient_accumulation_steps 1 \
# --max_seq_length 514 \
# --learning_rate 1e-6 \
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
# --per_device_train_batch_size 20 \
# --per_device_eval_batch_size 20
# # --max_train_samples 100 \
# # --max_eval_samples 10 \
# # --max_predict_samples 10

######################## TRAINING ON MEMES_EXIST 2024 DATASET ############

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=0 python3 src/train_multimodal_classifier.py \
--num_gcn_layers 1 \
--no_peft \
--image_caption caption_promptA \
--run_name clip-vit-large_batch64_5e-6lr_1gcn_xuinit_mami64bs_captionPA_soft_label \
--logging online \
--target_column soft_label_task4 \
--text_column text_en \
--image_column image \
--do_train \
--do_eval \
--do_predict \
--output_dir output_hard_label_task4 \
--checkpoint_path output_mami/clip-vit-large_batch64_5e-6lr_1gcn_xuinit \
--model_name_or_path openai/clip-vit-large-patch14 \
--dataset_name paoloitaliani/memes_exist2024 \
--dataset_subset default \
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
--remove_unused_columns \
--metric_for_best_model loss \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 64


PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=0 python3 src/train_multimodal_classifier.py \
--num_gcn_layers 1 \
--no_peft \
--image_caption caption_promptA \
--run_name clip-vit-large_batch100_5e-6lr_1gcn_xuinit_mami64bs_captionPA_soft_label \
--logging online \
--target_column soft_label_task4 \
--text_column text_en \
--image_column image \
--do_train \
--do_eval \
--do_predict \
--output_dir output_hard_label_task4 \
--checkpoint_path output_mami/clip-vit-large_batch64_5e-6lr_1gcn_xuinit \
--model_name_or_path openai/clip-vit-large-patch14 \
--dataset_name paoloitaliani/memes_exist2024 \
--dataset_subset default \
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
--remove_unused_columns \
--metric_for_best_model loss \
--per_device_train_batch_size 100 \
--per_device_eval_batch_size 100


PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=0 python3 src/train_multimodal_classifier.py \
--num_gcn_layers 1 \
--no_peft \
--image_caption caption_promptA \
--run_name clip-vit-large_batch64_5e-6lr_1gcn_xuinit_mami20bs_captionPA_soft_label \
--logging online \
--target_column soft_label_task4 \
--text_column text_en \
--image_column image \
--do_train \
--do_eval \
--do_predict \
--output_dir output_hard_label_task4 \
--checkpoint_path output_mami/clip-vit-large_batch20_5e-6lr_1gcn_xuinit \
--model_name_or_path openai/clip-vit-large-patch14 \
--dataset_name paoloitaliani/memes_exist2024 \
--dataset_subset default \
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
--remove_unused_columns \
--metric_for_best_model loss \
--per_device_train_batch_size 100 \
--per_device_eval_batch_size 100