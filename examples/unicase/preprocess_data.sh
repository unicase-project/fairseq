#!/bin/bash


# below script assume that you have at least 80 threads for processing

SPM_MODEL=$1

# create encoded file for test and dev
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_test.txt > en_test.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_valid.txt > en_valid.txt &

wait
echo "Completed SPM encoding of dev/test data"

# create one encoded file for each training shard
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0000.txt > en_head_0000.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0001.txt > en_head_0001.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0002.txt > en_head_0002.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0003.txt > en_head_0003.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0004.txt > en_head_0004.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0005.txt > en_head_0005.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0006.txt > en_head_0006.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0007.txt > en_head_0007.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0008.txt > en_head_0008.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0009.txt > en_head_0009.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0010.txt > en_head_0010.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0011.txt > en_head_0011.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0012.txt > en_head_0012.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0013.txt > en_head_0013.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0014.txt > en_head_0014.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0015.txt > en_head_0015.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0016.txt > en_head_0016.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0017.txt > en_head_0017.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0018.txt > en_head_0018.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0019.txt > en_head_0019.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0020.txt > en_head_0020.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0021.txt > en_head_0021.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0022.txt > en_head_0022.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0023.txt > en_head_0023.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0024.txt > en_head_0024.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0025.txt > en_head_0025.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0026.txt > en_head_0026.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0027.txt > en_head_0027.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0028.txt > en_head_0028.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0029.txt > en_head_0029.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0030.txt > en_head_0030.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0031.txt > en_head_0031.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0032.txt > en_head_0032.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0033.txt > en_head_0033.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0034.txt > en_head_0034.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0035.txt > en_head_0035.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0036.txt > en_head_0036.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0037.txt > en_head_0037.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0038.txt > en_head_0038.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0039.txt > en_head_0039.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0040.txt > en_head_0040.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0041.txt > en_head_0041.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0042.txt > en_head_0042.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0043.txt > en_head_0043.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0044.txt > en_head_0044.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0045.txt > en_head_0045.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0046.txt > en_head_0046.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0047.txt > en_head_0047.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0048.txt > en_head_0048.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0049.txt > en_head_0049.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0050.txt > en_head_0050.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0051.txt > en_head_0051.txt &
spm_encode --model=$MODEL_SPM --output_format=piece < ../data/raw_data/en_head_0052.txt > en_head_0052.txt &

wait
echo "Completed SPM encoding of train data"

cat en_head_0000.txt en_head_0001.txt en_head_0002.txt en_head_0003.txt en_head_0004.txt > en_train0.txt &
cat en_head_0005.txt en_head_0006.txt en_head_0007.txt en_head_0008.txt en_head_0009.txt > en_train1.txt &
cat en_head_0010.txt en_head_0011.txt en_head_0012.txt en_head_0013.txt en_head_0014.txt > en_train2.txt &
cat en_head_0015.txt en_head_0016.txt en_head_0017.txt en_head_0018.txt en_head_0019.txt > en_train3.txt &
cat en_head_0020.txt en_head_0021.txt en_head_0022.txt en_head_0023.txt en_head_0024.txt > en_train4.txt &
cat en_head_0025.txt en_head_0026.txt en_head_0027.txt en_head_0028.txt en_head_0029.txt > en_train5.txt &
cat en_head_0030.txt en_head_0031.txt en_head_0032.txt en_head_0033.txt en_head_0034.txt > en_train6.txt &
cat en_head_0035.txt en_head_0036.txt en_head_0037.txt en_head_0038.txt en_head_0039.txt > en_train7.txt &
cat en_head_0040.txt en_head_0041.txt en_head_0042.txt en_head_0043.txt en_head_0044.txt > en_train8.txt &
cat en_head_0045.txt en_head_0046.txt en_head_0047.txt en_head_0048.txt en_head_0049.txt > en_train9.txt &

wait

rm en_head_*
echo "Completed creation of training shards"

fairseq-preprocess --only-source --srcdict dict.txt --trainpref en_train0.txt --validpref en_valid.txt --testpref en_test.txt --destdir data-bin0/ --workers 8 &
fairseq-preprocess --only-source --srcdict dict.txt --trainpref en_train1.txt --validpref en_valid.txt --testpref en_test.txt --destdir data-bin1/ --workers 8 &
fairseq-preprocess --only-source --srcdict dict.txt --trainpref en_train2.txt --validpref en_valid.txt --testpref en_test.txt --destdir data-bin2/ --workers 8 &
fairseq-preprocess --only-source --srcdict dict.txt --trainpref en_train3.txt --validpref en_valid.txt --testpref en_test.txt --destdir data-bin3/ --workers 8 &
fairseq-preprocess --only-source --srcdict dict.txt --trainpref en_train4.txt --validpref en_valid.txt --testpref en_test.txt --destdir data-bin4/ --workers 8 &
fairseq-preprocess --only-source --srcdict dict.txt --trainpref en_train5.txt --validpref en_valid.txt --testpref en_test.txt --destdir data-bin5/ --workers 8 &
fairseq-preprocess --only-source --srcdict dict.txt --trainpref en_train6.txt --validpref en_valid.txt --testpref en_test.txt --destdir data-bin6/ --workers 8 &
fairseq-preprocess --only-source --srcdict dict.txt --trainpref en_train7.txt --validpref en_valid.txt --testpref en_test.txt --destdir data-bin7/ --workers 8 &
fairseq-preprocess --only-source --srcdict dict.txt --trainpref en_train8.txt --validpref en_valid.txt --testpref en_test.txt --destdir data-bin8/ --workers 8 &
fairseq-preprocess --only-source --srcdict dict.txt --trainpref en_train9.txt --validpref en_valid.txt --testpref en_test.txt --destdir data-bin9/ --workers 8 &

wait
echo "Completed fairseq-preprocess"
rm en_train*

echo "Clean-up completed"
