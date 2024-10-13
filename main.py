#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author:
# @Date  : 2023/9/23 15:25
# @Desc  :
import argparse
import os
import random
import time

import numpy as np
import torch
from loguru import logger
from model import MBLFE

from data_set import DataSet

from trainer import Trainer


seed = 2021
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # True can improve train speed
    torch.backends.cudnn.deterministic = True  # Guarantee that the convolution algorithm returned each time will be deterministic
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Set args', add_help=False)

    parser.add_argument('--embedding_size', type=int, default=64, help='')
    parser.add_argument('--reg_weight', type=float, default=1e-3, help='')
    parser.add_argument('--layers', type=int, default=2)

    parser.add_argument('--data_name', type=str, default='tmall', help='')
    parser.add_argument('--behaviors', help='', action='append')
    parser.add_argument('--loss_type', type=str, default='bpr', help='')
    parser.add_argument('--neg_count', type=int, default=4)

    parser.add_argument('--label_size', type=int, default=64)
    parser.add_argument('--num_experts', type=int, default=10)
    parser.add_argument('--irr_sample_no', type=int, default=5)
    parser.add_argument('--ssl_tau', type=float, default=0.7)
    parser.add_argument('--ssl_reg', type=float, default=0.1)


    parser.add_argument('--if_load_model', type=bool, default=False, help='')
    parser.add_argument('--gpu_no', type=int, default=1, help='')
    parser.add_argument('--topk', type=list, default=[10, 20, 50, 80], help='')
    parser.add_argument('--metrics', type=list, default=['hit', 'ndcg'], help='')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--decay', type=float, default=0., help='')
    parser.add_argument('--batch_size', type=int, default=1024, help='')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='')
    parser.add_argument('--min_epoch', type=str, default=5, help='')
    parser.add_argument('--epochs', type=str, default=500, help='')
    parser.add_argument('--model_path', type=str, default='./check_point', help='')
    parser.add_argument('--check_point', type=str, default='', help='')
    parser.add_argument('--model_name', type=str, default='tmall', help='')
    parser.add_argument('--log_name', type=str, default='tmall', help='')
    parser.add_argument('--train_sample_path', type=str, default='samples')
    parser.add_argument('--device', type=str, default='cuda:0', help='')

    args = parser.parse_args()
    if args.data_name == 'tmall':
        args.data_path = './data/Tmall'
        args.behaviors = ['click', 'collect', 'cart', 'buy']
    elif args.data_name == 'yelp':
        args.data_path = './data/Yelp'
        args.behaviors = ['tip', 'neutral', 'neg', 'pos']
    elif args.data_name == 'taobao':
        args.data_path = './data/taobao'
        args.behaviors = ['view', 'cart', 'buy']

    else:
        raise Exception('data_name cannot be None')

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # args.device = device

    TIME = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
    args.TIME = TIME

    logfile = '{}_enb_{}_{}'.format(args.data_name, args.embedding_size, TIME)
    # args.train_writer = SummaryWriter('./log/train/' + logfile)
    # args.test_writer = SummaryWriter('./log/test/' + logfile)
    logger.add('./log/{}/{}.log'.format(args.log_name, logfile), encoding='utf-8')

    start = time.time()
    dataset = DataSet(args)
    path = os.path.join(args.data_path, args.train_sample_path)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # if not os.path.exists(path):
    #     os.makedirs(path)
    #     dataset.generate_train_samples(path)
    #     logger.info('train_samples generate complete !!!')

    model = MBLFE(args, dataset).to(args.device)
    # modules = __import__(args.model_name)
    # model = getattr(modules, 'MBLFE')(args, dataset).to(args.device)

    trainer = Trainer(model, dataset, args)

    logger.info(args.__str__())
    logger.info(model)

    trainer.train_model()
    # trainer.evaluate(0, 10, dataset.test_dataset(), dataset.test_interacts, dataset.test_gt_length, dataset.valid_mask)
    logger.info('train end total cost time: {}'.format(time.time() - start))



