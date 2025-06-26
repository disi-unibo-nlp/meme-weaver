CUDA_VISIBLE_DEVICES=3 python3 evaluation/affinity_matrix_stats_multimodal.py \
--percentile 0 \
--fig_type top \
--output_dir output_hard_label_task4 \
--batch_size 64 \
--run_name clip-vit-large_batch64_5e-6lr_1gcn_xuinit_mami20bs_captionPA_concat