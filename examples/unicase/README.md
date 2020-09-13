

### 1) Preprocess the data

Data is created based on cc-net english dataset. Data should be extracted from `gzip` files 
to `txt` files so it can be easilyprocessed by fairseq binarizer.

#### Data source
```
python ccnet_to_raw_text.py --inp_dir /data-c/shared/corpora/ccnet/en --out_dir /data-c/shared/ar/unicase/data/raw_data --num_work 8
```
We should create set for test and validation.
```shell script
head -50000 en_head_0053.txt > en_test.txt
tail -50000 en_head_0053.txt > en_valid.txt
```

#### Tokenizer

Create lowercased tokenizer
```shell script
spm_train --input=../data/lowercased/en_head_0004.txt  --unk_id=0 --bos_id=-1 --eos_id=-1 \
--pad_id=-1 --vocab_size=25000 --model_prefix=lower --character_coverage 0.9999 \
--num_sub_iterations 2 --num_threads 32 --input_sentence_size 8000000 --shuffle_input_sentence=true
```

And convert it to unicase-style SPM tokenizer
```shell script
python <unicase_utils_path>/convert_to_unicase_spm.py --spm_lower_model lower.model --out_prefix unicase
```


#### Create binarized data using spm and fairseq
```shell script
<unicase_utils_path>/preprocess_data.sh unicase.model
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
