

### 1) Preprocess the data

Data is created based on cc-net english dataset. Data should be extracted from `gzip` files 
to `txt` files so it can be easilyprocessed by fairseq binarizer.

#### Data extraction
```
python ccnet_to_raw_text.py --inp_dir /data-c/shared/corpora/ccnet/en --out_dir /data-c/shared/ar/unicase/data/raw_data --num_work 8
```

We should create set for test and validation.

```shell script
head -50000 en_head_0053.txt > en_test.txt
tail -50000 en_head_0053.txt > en_valid.txt
```


# create lowercased tokenizer

spm_train --input=../data/lowercased/en_head_0004.txt  --unk_id=0 --bos_id=-1 --eos_id=-1 \
--pad_id=-1 --vocab_size=25000 --model_prefix=lower --character_coverage 0.9999 \
--num_sub_iterations 2 --num_threads 32 --input_sentence_size 8000000 --shuffle_input_sentence=true




Then use fairseq and SPM to prepare binarized data
```shell script
spm_encode --model=unigram32.model --output_format=piece < ../data/raw_data/en_test.txt > en_test.txt &
spm_encode --model=unigram32.model --output_format=piece < ../data/raw_data/en_valid.txt > en_valid.txt &

# for each file
spm_encode --model=unigram32.model --output_format=piece < ../data/raw_data/en_head_0000.txt > en_head_0000.txt &
```

Then join data to one file:
```shell script
cat en_head_0000.txt en_head_0001.txt ... > en_train.txt
```

And run processing by fairseq

```shell script
fairseq-preprocess \
    --only-source \
    --srcdict dict.txt \
    --trainpref en_train.txt \
    --validpref en_valid.txt \
    --testpref en_test.txt \
    --destdir data-bin/ \
    --workers 60
```


Now we are ready to train model. Below example is on dgxA100

# xlm roberta on dgx a100
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;
TOTAL_UPDATES=500000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0005          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=32        # Number of sequences per batch (batch size)
UPDATE_FREQ=8           # Increase the batch size
SAVE_INTERVAL_UPDATES=25000
DATA_DIR=data-bin0:data-bin1:data-bin2:data-bin3:data-bin4:data-bin5:data-bin6:data-bin7:data-bin8:data-bin9
fairseq-train --fp16 $DATA_DIR \
        --task masked_lm --criterion masked_lm --bpe sentencepiece \
        --arch xlmr_base --sample-break-mode complete_doc --tokens-per-sample $TOKENS_PER_SAMPLE \
        --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
        --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
        --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
        --max-update $TOTAL_UPDATES --log-format tqdm --skip-invalid-size-inputs-valid-test \
        --save-interval-updates $SAVE_INTERVAL_UPDATES --validate-interval-updates $SAVE_INTERVAL_UPDATES --save-interval 9999 --validate-interval 9999  \

# unicase on a100
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;
TOTAL_UPDATES=500000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0005          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=32        # Number of sequences per batch (batch size)
UPDATE_FREQ=8           # Increase the batch size
SAVE_INTERVAL_UPDATES=25000
DATA_DIR=data-bin0:data-bin1:data-bin2:data-bin3:data-bin4:data-bin5:data-bin6:data-bin7:data-bin8:data-bin9
fairseq-train --fp16 $DATA_DIR \
        --task masked_lm --criterion unicase_masked_lm --bpe sentencepiece \
        --arch unicase_base --sample-break-mode complete_doc --tokens-per-sample $TOKENS_PER_SAMPLE \
        --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
        --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
        --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
        --max-update $TOTAL_UPDATES --log-format tqdm --skip-invalid-size-inputs-valid-test \
        --save-interval-updates $SAVE_INTERVAL_UPDATES --validate-interval-updates $SAVE_INTERVAL_UPDATES \
        --save-interval 9999 --validate-interval 9999 --dict_cased_words 68847 --dict_non_cased_words 2159 \
