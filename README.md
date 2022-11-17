# Multi-modal Dialogue Scene & Session Identification

## Dataset

### Statistics

|       | clips | utterances | scenes | sessions |
| ----- | ----- | ---------- | ------ | -------- |
| train | 8,025 | 200,073    | 11,440 | 20,996   |
| valid | 416   | 10,532     | 615    | 1,270    |
| test  | 403   | 10,260     | -      | -        |

### Download

The features can be find [HERE](https://pan.baidu.com/s/1DnCerijBJ3Q68eVBQEGMLw?pwd=fsmt)

### Baseline

![baseline](pics/baseline.png?)

### Instruction

unzip features to inputs/features/resnet

```python
python run.py \
--train_path inputs/preprocessed/MDSS_train.json \
--valid_path inputs/preprocessed/MDSS_valid.json \
--test_path inputs/preprocessed/MDSS_test.json \
--train_batch_size 8 \
--lr 1e-5 \
--gradient_accumulation_steps 2\
--model_checkpoint bert-base-uncased \
--ft 1 \ # 1: fitune 0: train from scratch
--exp_set _baseline \
--video 0 \ # 0: session identification 1: scene identification
--gpuid 0 \
--test_each_epoch 1 \ # test for each epoch

```

Following evaluation/submission.json, please merge the result file in results/ by yourself. 🤗️

### Results

| Task                   | F1   |
| ---------------------- | ---- |
| Scene Identification   | 30.7 |
| Session Identitication | 35.4 |
