# ################# MAMI #####################
######## CLIP(MemeWeaver) accuracy: 0.752 ± 0.018 ##########
python3 evaluation/emb_comparison.py \
--output_dir output_mami \
--dataset mami_hf_repo \
--run_name clip-vit-large_batch20_5e-6lr_1gcn_xuinit_mfb \
--id_column file_name \
--batch_size 40 \
--target_column label

######## CLIP accuracy: 0.747 ± 0.019 ##########
python3 evaluation/emb_comparison.py \
--output_dir output_mami \
--dataset mami_hf_repo \
--run_name clip-vit-large_batch20_5e-6lr_mfb \
--id_column file_name \
--batch_size 20 \
--target_column label


################# EXIST #####################
######## CLIP accuracy: 0.721 ± 0.047 ##########
# python3 evaluation/emb_comparison.py \
# --output_dir output_hard_label_task4 \
# --dataset exist_hf_repo \
# --run_name clip-vit-large_batch64_5e-6lr_mami20bs_captionPA_concat \
# --id_column id \
# --batch_size 64 \
# --target_column hard_label_task4

######## CLIP(MemeWeaver) accuracy: 0.747 ± 0.025 ##########
# python3 evaluation/emb_comparison.py \
# --output_dir output_hard_label_task4 \
# --dataset exist_hf_repo \
# --run_name clip-vit-large_batch64_5e-6lr_1gcn_xuinit_mami20bs_captionPA_concat \
# --id_column id \
# --batch_size 27 \
# --target_column hard_label_task4