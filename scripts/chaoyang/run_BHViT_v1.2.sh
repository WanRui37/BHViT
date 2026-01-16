source set_mirror.sh

DATA_DIR=/mnt/data/small_dataset/chaoyang/

version="v1.2"
output_dir="result/chaoyang/w1a1_train_$version"

# 检查目录是否存在，不存在则创建
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=25651 main_new.py \
    --data-set=chaoyang \
    --num-workers=1 \
    --batch-size=32 \
    --epochs=300 \
    --dropout=0.0 \
    --drop-path=0.0 \
    --opt=adamw \
    --sched=cosine \
    --weight-decay=1e-3 \
    --lr=4e-3 \
    --warmup-epochs=0 \
    --color-jitter=0.0 \
    --aa=noaug \
    --reprob=0.0 \
    --mixup=0.0 \
    --cutmix=0.0 \
    --data-path=${DATA_DIR} \
    --output-dir=${output_dir} \
    --teacher-model-type=deit \
    --teacher-model=configs/deit-small-patch16-224 \
    --teacher-model-file=/mnt/wr/3-LLM/9-ViT/GSB-Vision-Transformer/teacher/chaoyang/teacher.pth \
    --model=configs/BHViT \
    --model-type=BHVIT \
    --replace-ln-bn \
    --weight-bits=1 \
    --input-bits=1 \
    --shift3 \
    --shift5 \
    --some-fp \
    --resume= \
    2>&1 | tee "$output_dir/train.log"
    #--current-best-model= \
    #--recu \
    #--regularization_loss \
