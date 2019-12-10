DATA=../data_bin 

CUDA_VISIBLE_DEVICES=0 python3 ../fairseq-master/train.py $DATA \
    -s en \
    -t de \
    --lr 0.0007 --min-lr 1e-09 --weight-decay 0.0 --clip-norm 0.0 \
    --dropout 0.1 \
    --max-tokens 512 \
    --update-freq 1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --save-dir checkpoints/ \
    --tensorboard-logdir tensorboard/logs \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --log-format simple \
    --log-interval 100 \
    --save-interval-updates 2000 \
    --max-update 150000 \
    --max-epoch 100 \
    --encoder-normalize-before 

    
