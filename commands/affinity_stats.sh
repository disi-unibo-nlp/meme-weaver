CUDA_VISIBLE_DEVICES=0 python3 evaluation/affinity_matrix_stats.py \
--percentile 90 \
--fig_type top \
--output_dir output_mami \
--run_name xlm-roberta-large_batch140_10eps_1gcn 

CUDA_VISIBLE_DEVICES=0 python3 evaluation/affinity_matrix_stats.py \
--percentile 10 \
--fig_type bottom \
--output_dir output_mami \
--run_name xlm-roberta-large_batch140_10eps_1gcn 