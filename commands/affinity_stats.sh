CUDA_VISIBLE_DEVICES=0 python3 evaluation/affinity_matrix_stats_multimodal.py \
--percentile 0 \
--fig_type top \
--output_dir output_mami \
--batch_size 40 \
--run_name clip-vit-large_batch20_5e-6lr_1gcn_xuinit_mfb