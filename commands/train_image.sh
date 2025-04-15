PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=2 python3 src/train_image_classifier.py \
--no_peft \
--run_name swin-tiny_batch100 \
--logging online \
--target_column label \
--do_train \
--do_eval \
--do_predict \
--output_dir output_mami \
--model_name_or_path microsoft/swin-tiny-patch4-window7-224 \
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
# --max_train_samples 200 \
# --max_eval_samples 200 \
# --max_predict_samples 200 
