#! /bin/bash

split=0
seeds=(1 11 65537 101 1999 2017)
seed=${seeds[$split]}

DATA=$1
SAVE_ROOT=$2
DEVICES=$3
model=transformer_lra_imdb
# exp_name=1_apollo_luna_k16_run${seed}

SAVE=${SAVE_ROOT}
# rm -rf ${SAVE}
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh

CUDA_VISIBLE_DEVICES=0,1 python -u train.py ${DATA} \
    --seed $RANDOM --ddp-backend c10d --fp16 \
    --apply-bert-init \
    --dropout 0.3 --attention-dropout 0.3 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    -a ${model} --task transformer_lra \
    --optimizer adam --lr 0.00005 --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --batch-size 16 --sentence-avg --update-freq 1 \
    --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion lra_cross_entropy --max-update 20000 --save-interval-updates 100 \
    --warmup-updates 8000 --warmup-init-lr '1e-07' \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 | tee ${SAVE}/log.txt

date
wait

# opt=${SAVE}/test_best.log
# python -u fairseq_cli/generate.py $DATA --gen-subset test -s $src -t $tgt --path ${SAVE}/checkpoint_best.pt --batch-size 300 --remove-bpe "@@ " --beam 5 --max-len-a 2 --max-len-b 0 --quiet | tee ${opt}

# opt=${SAVE}/test_last.log
# python -u fairseq_cli/generate.py $DATA --gen-subset test -s $src -t $tgt --path ${SAVE}/checkpoint_last.pt --batch-size 300 --remove-bpe "@@ " --beam 5 --max-len-a 2 --max-len-b 0 --quiet | tee ${opt}

# python scripts/average_checkpoints.py --inputs ${SAVE} --output ${SAVE}/checkpoint_last10.pt --num-epoch-checkpoints 10
# rm -f ${SAVE}/checkpoint2*.pt
# rm -f ${SAVE}/checkpoint_254_500000.pt

# opt=${SAVE}/test_last10.log
# python -u fairseq_cli/generate.py $DATA --gen-subset test -s $src -t $tgt --path ${SAVE}/checkpoint_last10.pt --batch-size 300 --remove-bpe "@@ " --beam 5 --max-len-a 2 --max-len-b 0 --quiet | tee ${opt}