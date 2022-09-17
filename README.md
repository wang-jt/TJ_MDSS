# Multi-modal Dialogue Scene & Session Identification

## Dataset

### Statistics

|       | clips | utterances | scenes | sessions |
| ----- | ----- | ---------- | ------ | -------- |
| train | 8,025 | 200,073    | 11,440 | 20,996   |
| valid | 416   | 10,532     | 615    | 1,270    |
| test  | 403   | 10,260     | -      | -        |

### Download

The features can be find [HERE]()

## Instructions

unzip features to inputs/features/resnet

```python
python train_resnetnsp.py \
--train_path inputs/preprocessed/MDSS_train.json \
--valid_path inputs/preprocessed/MDSS_valid.json \
--test_path inputs/preprocessed/MDSS_test.json \
--train_batch_size 8 \
--lr 1e-5 \
--gradient_accumulation_steps 2\
--model_checkpoint bert-base-uncased \
--ft 1 \ # 0: fitune 1: train from scratch
--exp_set _baseline \
--video 0 \ # 0: session identification 1: scene identification
--gpuid 0 \
--test 1 \ # test for each epoch

```
