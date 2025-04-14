CUDA_VISIBLE_DEVICES=2 python3 evaluation/affinity_matrix_stats.py \
--percentile 10 \
--fig_type bottom \
--output_dir output_mami \
--batch_size 100 \
--run_name xlm-roberta-large_batch100_10eps_reprod_1gcn_seed45