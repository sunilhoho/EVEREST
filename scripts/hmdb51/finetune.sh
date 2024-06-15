OUTPUT_DIR='output/hmdb51/ft'
DATA_PATH='path_to_data/HMDB51/' # path to HMDB51 annotation file (train.csv/val.csv/test.csv)
MODEL_PATH='output/hmdb51/pt/checkpoint-4799.pth'

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port 12320  run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --data_set HMDB51 \
    --nb_classes 51 \
    --batch_size 16 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 20 \
    --num_frames 16 \
    --sampling_rate 2 \
    --num_sample 1 \
    --opt adamw \
    --lr 1e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 50 \
    --test_num_segment 10 \
    --test_num_crop 3 \
    --use_checkpoint \
    --dist_eval \
    --enable_deepspeed \
    --mcm \
    --mcm_ratio 0.4 \
    --mixup 0.0 \
    --cutmix 0.0
