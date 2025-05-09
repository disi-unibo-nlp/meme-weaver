CUDA_VISIBLE_DEVICES=2 python3 evaluation/affinity_matrix_stats_multimodal.py \
--percentile 0 \
--fig_type top \
--output_dir output_mami \
--batch_size 20 \
--run_name clip-vit-large_batch64_5e-6lr_1gcn_xuinit