#! /bin/bash

split=0
seeds=(1 11 65537 101 1999 2017)
seed=${seeds[$split]}

DATA=$1
SAVE_ROOT=$2
DEVICES=$3
model=transformer_lra_imdb_eff
# exp_name=1_apollo_luna_k16_run${seed}

SAVE=${SAVE_ROOT}
# rm -rf ${SAVE}
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh

CUDA_VISIBLE_DEVICES=0,1 python -u train.py ${DATA} \
    --seed $RANDOM --ddp-backend c10d \
    --apply-bert-init \
    --dropout 0.3 --attention-dropout 0.3 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    -a ${model} --task transformer_lra \
    --optimizer adam --lr 0.00005 --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --batch-size 2 --sentence-avg --update-freq 1 \
    --no-save --max-epoch 1 --disable-validation \
    --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion lra_cross_entropy --max-update 1000 --save-interval-updates 26000 \
    --warmup-updates 8000 --warmup-init-lr '1e-07' \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 | tee ${SAVE}/log.txt