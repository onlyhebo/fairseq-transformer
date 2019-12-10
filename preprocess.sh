TEXT=../raw_data 
python3 ../fairseq-master/preprocess.py \
    --source-lang en \
    --target-lang de \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --testpref $TEXT/test \
    --destdir ../data_bin
