PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=1 python3 src/train_multimodal_classifier.py \
--num_gcn_layers 1 \
--no_peft \
--run_name clip-vit-large_batch100_5e-6lr_1gcn_xuinit \
--logging online \
--target_column label \
--text_column text \
--image_column image_path \
--do_train \
--do_eval \
--do_predict \
--output_dir output_mami \
--model_name_or_path openai/clip-vit-large-patch14 \
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
--per_device_train_batch_size 100 \
--per_device_eval_batch_size 100 
# --max_train_samples 100 \
# --max_eval_samples 10 \
# --max_predict_samples 10 

# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=3 python3 src/train_multimodal_classifier.py \
# --num_gcn_layers 2 \
# --custom_gcn learn_upd \
# --no_peft \
# --run_name clip-vit-large_batch100_5e-6lr_2gcn \
# --logging online \
# --target_column label \
# --text_column text \
# --image_column image_path \
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
# --per_device_train_batch_size 100 \
# --per_device_eval_batch_size 100 
# # --max_train_samples 100 \
# # --max_eval_samples 10 \
# # --max_predict_samples 10 

# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=3 python3 src/train_multimodal_classifier.py \
# --apply_ffw \
# --num_gcn_layers 1 \
# --no_peft \
# --run_name clip-vit-large_batch100_5e-6lr_1gcn_ffw \
# --logging online \
# --target_column label \
# --text_column text \
# --image_column image_path \
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
# --per_device_train_batch_size 100 \
# --per_device_eval_batch_size 100 
# # --max_train_samples 100 \
# # --max_eval_samples 10 \
# # --max_predict_samples 10 
