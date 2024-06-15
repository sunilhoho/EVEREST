OUTPUT_DIR='output/hmdb51/pt'
DATA_PATH='path_to_data/HMDB51/train.csv'

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
         --master_port 12320 run_ms_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type motion-centric \
        --motion_centric_masking_ratio 0.7 \
        --mask_ratio 0.9 \
        --model pretrain_videoms_base_patch16_224 \
        --decoder_depth 4 \
        --lr 1e-3 \
        --batch_size 24 \
        --num_frames 16 \
        --sampling_rate 2 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --epochs 4800 \
        --save_ckpt_freq 100 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}
