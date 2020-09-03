

### 1) Preprocess the data

Data is created based on cc-net english dataset. Data should be extracted from `gzip` files 
to `txt` files so it can be easilyprocessed by fairseq binarizer.

#### Data extraction
```shell script
python ccnet_to_raw_text.py --inp_dir /home/rpowalski/python/unicaselm/ --out_dir /home/rpowalski/python/unicaselm/ --num_workers 8

```
