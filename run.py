# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import sys
import math
import time
import random
import copy
import json
import logging
import datetime
import collections
from pprint import pformat
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine, Checkpoint, DiskSaver
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup
from ignite.handlers.stores import EpochOutputStore
from ignite.metrics import RunningAverage, Average
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler

from transformers import AdamW, BertTokenizer
from transformers.file_utils import (CONFIG_NAME, WEIGHTS_NAME)
from transformers.models.bert.configuration_bert import BertConfig
from transformers.utils.dummy_pt_objects import Adafactor

from model.resnetnspBert import BertForSegClassification, BertForVisSegClassification
from utils.metrics import F1ScoreMetric
from data.seg_resnetnsp_dataset import DataSet, collate_fn, get_dataset

logger = logging.getLogger(__file__)


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()

def get_data_loaders_new(args, tokenizer):
    # trian data
    print('start load train data.')
    train_data = get_dataset(tokenizer, args.train_path)
    # valid data
    print('start load train data.')
    valid_data = get_dataset(tokenizer, args.valid_path)
    # test data
    print('start load train data.')
    test_data = get_dataset(tokenizer, args.test_path)
    
    if args.video: 
        with open(args.feature_path) as jh:
            feature = json.load(jh)
        train_dataset = DataSet(train_data, tokenizer, feature)
        valid_dataset = DataSet(valid_data, tokenizer, feature)
        test_dataset = DataSet(test_data, tokenizer, feature)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=2, shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=True))
        valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, num_workers=0, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=True))
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=0, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=True))
    else:
        train_dataset = DataSet(train_data, tokenizer, None)
        valid_dataset = DataSet(valid_data, tokenizer, None)
        test_dataset = DataSet(test_data, tokenizer, None)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=2, shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=None))
        valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, num_workers=0, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=None))
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=0, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=None))
    return train_loader, valid_loader, test_loader

def train():
    parser = ArgumentParser()
    parser.add_argument("--train_path", type=str, default="inputs/preprocessed/MDSS_train.json", help="Path of the trainset")
    parser.add_argument("--valid_path", type=str, default="inputs/preprocessed/MDSS_valid.json", help="Path of the validset")
    parser.add_argument("--test_path", type=str, default="inputs/preprocessed/MDSS_test.json", help="Path of the testset")
    parser.add_argument("--feature_path", type=str, default="inputs/MDSS_clipid2frames.json", help="Path of the feature")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=2, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for testing")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--gpuid", type=str, default='', help='select proper gpu id')
    parser.add_argument("--model", type=str, default='bert', help='Pretrained Model name')
    parser.add_argument('--video', type=int, default=1, help='if use video: 1 use 0 not')
    parser.add_argument('--exp_set', type=str, default='_test')
    parser.add_argument('--model_checkpoint', type=str, default="/share2/wangyx/prev_trained_model/bert-base-uncased")

    parser.add_argument('--test_each_epoch', type=int, default=1, choices=[0, 1])
    parser.add_argument('--ft', type=int, default=1, choices=[0, 1], help='1: finetune bert 0: train from scratch')
    parser.add_argument('--warmup_init', type=float, default=1e-07)
    parser.add_argument('--warmup_duration', type=float, default=5000)
    args = parser.parse_args()

    args.valid_batch_size = args.train_batch_size
    args.test_batch_size = args.train_batch_size
    args.model = 'bert'
    exp_set = args.exp_set
    args.exp = args.model + exp_set
    args.log_path = 'ckpts/' + args.exp + '/'
    args.tb_path = 'tb_logs/' + args.exp + '/'
    if args.device == 'cuda':
        args.device = 'cuda:' + args.gpuid

    # select model
    if args.model == 'bert':
        args.model_checkpoint = args.model_checkpoint
    else:
        raise ValueError('NO IMPLEMENTED MODEL!')

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.tb_path):
        os.makedirs(args.tb_path)
    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: %s", pformat(args))
    
    tokenizer_class = BertTokenizer
    if args.video:
        model_class = BertForVisSegClassification
        if not args.ft:
            bert_config = BertConfig()
            model = model_class(bert_config)
        else:
            model = model_class.from_pretrained(args.model_checkpoint)
    else:
        model_class = BertForSegClassification
        if not args.ft:
            bert_config = BertConfig()
            model = model_class(bert_config)
        else:
            model = model_class.from_pretrained(args.model_checkpoint)
    
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    args.pad = model.config.pad_token_id

    logger.info("Prepare datasets for Bart")
    train_loader, valid_loader, test_loader = get_data_loaders_new(args, tokenizer)

    
    # Training function and trainer
    def update(engine, batch):
        dialog_ids, dialog_type_ids, dialog_mask, session_label_ids, session_indexs, \
        feature_ids, feature_type_ids, feature_mask, scene_label_ids, scene_indexs, utter_lst, vid_lst = batch
        if args.video == 0:   
            dialog_ids = dialog_ids.to(args.device)
            dialog_type_ids = dialog_type_ids.to(args.device)
            dialog_mask = dialog_mask.to(args.device)
            session_label_ids = session_label_ids.to(args.device)
            session_indexs = [sess.to(args.device) for sess in session_indexs]
        else:
            feature_ids = feature_ids.to(args.device)
            feature_type_ids = feature_type_ids.to(args.device)
            feature_mask = feature_mask.to(args.device)
            scene_indexs = [scene.to(args.device) for scene in scene_indexs]
            scene_label_ids = scene_label_ids.to(args.device)

        # optimize Bert
        model.train(True)
        if args.video == 0:
            bsz = 16
            loss = model(dialog_ids[:bsz], dialog_mask[:bsz], dialog_type_ids[:bsz], \
                labels=session_label_ids[:bsz], seg_indexs=session_indexs[:bsz])[0]
        else:
            loss = model(feature_ids, feature_mask, feature_type_ids, labels=scene_label_ids, seg_indexs=scene_indexs)[0]
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)

    def valid(engine, batch):
        model.train(False)
        f1_metric = F1ScoreMetric(average='micro', num_classes=1, multiclass=False, threshold=0.5)
        f1_metric = f1_metric.to(args.device)
        with torch.no_grad():
            dialog_ids, dialog_type_ids, dialog_mask, session_label_ids, session_indexs, \
            feature_ids, feature_type_ids, feature_mask, scene_label_ids, scene_indexs, utter_lst, vid_lst = batch
            if args.video == 0:   
                dialog_ids = dialog_ids.to(args.device)
                dialog_type_ids = dialog_type_ids.to(args.device)
                dialog_mask = dialog_mask.to(args.device)
                session_indexs = [sess.to(args.device) for sess in session_indexs]
                label = session_label_ids.to(args.device)
                logits = model(dialog_ids, dialog_mask, dialog_type_ids, seg_indexs=session_indexs)[0]
            else:
                feature_ids = feature_ids.to(args.device)
                feature_type_ids = feature_type_ids.to(args.device)
                feature_mask = feature_mask.to(args.device)
                scene_indexs = [scene.to(args.device) for scene in scene_indexs]
                label = scene_label_ids.to(args.device)
                logits = model(feature_ids, feature_mask, feature_type_ids, seg_indexs=scene_indexs)[0]
            prob = F.softmax(logits, dim=1)
            f1_metric.update(prob[:, 1], label)
            f1 = f1_metric.compute()
        return f1.item()

    trainer = Engine(update)
    validator = Engine(valid)

              
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: validator.run(valid_loader))

    if args.ft:
        scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    else:
        torch_lr_scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader) - args.warmup_duration, 0.0)])
        scheduler = create_lr_scheduler_with_warmup(torch_lr_scheduler, warmup_start_value=args.warmup_init, warmup_duration=args.warmup_duration)
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # # Prepare metrics - note how we compute distributed metrics 
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    RunningAverage(output_transform=lambda x: x).attach(validator, "f1")

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=["loss"])
    val_pbar = ProgressBar(persist=True)
    val_pbar.attach(validator, metric_names=['f1'])

    tb_logger = TensorboardLogger(log_dir=args.tb_path)
    tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
    tb_logger.attach(validator, log_handler=OutputHandler(tag='validation', metric_names=["f1"], global_step_transform=global_step_from_engine(trainer)), event_name=Events.EPOCH_COMPLETED)

    checkpoint_handler = ModelCheckpoint(args.log_path, 'checkpoint', n_saved=args.n_epochs ,require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation
    
    torch.save(args, args.log_path + 'model_training_args.bin')
    getattr(model, 'module', model).config.to_json_file(os.path.join(args.log_path, CONFIG_NAME))
    tokenizer.save_vocabulary(args.log_path)

    best_score = Checkpoint.get_default_score_fn('f1')
    best_model_handler = Checkpoint(
        {'mymodel': getattr(model, 'module', model)},
        filename_prefix='best',
        save_handler=DiskSaver(args.log_path, create_dir=True, require_empty=False),
        score_name='f1',
        score_function=best_score,
        global_step_transform=global_step_from_engine(trainer, Events.ITERATION_COMPLETED),
        filename_pattern='{filename_prefix}_{global_step}_{score_name}={score}.{ext}'
    )
    validator.add_event_handler(Events.COMPLETED, best_model_handler)
    
    if args.test_each_epoch:
        @trainer.on(Events.EPOCH_COMPLETED)
        def test():
            model.train(False)  
            result = collections.defaultdict(list)
            with torch.no_grad():
                for batch in tqdm(test_loader, desc='test'):
                    dialog_ids, dialog_type_ids, dialog_mask, session_label_ids, session_indexs,\
                    feature_ids, feature_type_ids, feature_mask, scene_label_ids, scene_indexs, utter_lst, vid_lst = batch
                    if args.video == 0:   
                        dialog_ids = dialog_ids.to(args.device)
                        dialog_type_ids = dialog_type_ids.to(args.device)
                        dialog_mask = dialog_mask.to(args.device)
                        session_indexs = [sess.to(args.device) for sess in session_indexs]
                        logits = model(dialog_ids, dialog_mask, dialog_type_ids, seg_indexs=session_indexs)[0]
                        probs = F.softmax(logits, dim=1)
                        preds = torch.argmax(probs, dim=1)
                        for vid, pre in zip(vid_lst, preds):
                            result[vid].append(pre.item())

                    else:
                        feature_ids = feature_ids.to(args.device)
                        feature_type_ids = feature_type_ids.to(args.device)
                        feature_mask = feature_mask.to(args.device)
                        scene_indexs = [scene.to(args.device) for scene in scene_indexs]
                        logits = model(feature_ids, feature_mask, feature_type_ids, seg_indexs=scene_indexs)[0]
                        probs = F.softmax(logits, dim=1)
                        preds = torch.argmax(probs, dim=1)
                        for vid, pre in zip(vid_lst, preds):
                            result[vid].append(pre.item())
                        

            output_dir = 'results/{}/'.format(args.exp)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)        
            if args.video:
                with open(os.path.join(output_dir, 'scene_res_{}.json'.format(trainer.state.epoch)), 'w') as jh:
                    json.dump(result, jh)
            else:
                with open(os.path.join(output_dir, 'session_res_{}.json'.format(trainer.state.epoch)), 'w') as jh:
                    json.dump(result, jh)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        tb_logger.close()

if __name__ == "__main__":
    train()
