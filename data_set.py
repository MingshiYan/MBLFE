#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_set.py
# @Author:
# @Date  : 2021/11/1 11:38
# @Desc  :
import argparse
import os
import random
import json
from collections import defaultdict

import torch
import scipy.sparse as sp

from torch.utils.data import Dataset, DataLoader
import numpy as np

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class TestDate(Dataset):
    def __init__(self, user_count, item_count, samples=None):
        self.user_count = user_count
        self.item_count = item_count
        self.samples = samples

    def __getitem__(self, idx):
        return int(self.samples[idx])

    def __len__(self):
        return len(self.samples)

# class TrainDate(Dataset):
#     def __init__(self, train_data):
#         self.train_data = train_data
#
#     def __getitem__(self, idx):
#         return np.array(self.train_data[idx])
#
#     def __len__(self):
#         return len(self.train_data)


class TrainDate(Dataset):
    def __init__(self, user_count, item_count, neg_count, behavior_dict=None, behaviors=None):
        self.user_count = user_count
        self.item_count = item_count
        self.behavior_dict = behavior_dict
        self.behaviors = behaviors
        self.neg_count = neg_count

    def __getitem__(self, idx):
        # generate positive and negative samples pairs under each behavior
        total = []

        all_inter = self.behavior_dict['all'].get(str(idx + 1), None)
        for index, behavior in enumerate(self.behaviors):
            tmp = []
            items = self.behavior_dict[behavior].get(str(idx + 1), None)
            if items is None:
                tmp = [0] * (self.neg_count + 3 + self.neg_count + 1)  # (self.neg_count + 1) is the ground truth label
            else:
                tmp.append(idx + 1)
                pos = random.sample(items, 1)[0]
                tmp.append(pos)
                for i in range(self.neg_count):
                    neg = random.randint(1, self.item_count)
                    while np.isin(neg, all_inter):
                        neg = random.randint(1, self.item_count)
                    tmp.append(neg)
                tmp.append(index)
                tmp.append(1)
                tmp.extend([0] * self.neg_count)
            total.append(tmp)

        return np.array(total)

    def __len__(self):
        return self.user_count

class TrainSample(Dataset):
    def __init__(self, user_count, item_count, pos_sampling, neg_count, behavior_dict=None, behaviors=None):
        self.user_count = user_count
        self.item_count = item_count
        self.pos_sampling = pos_sampling
        self.behavior_dict = behavior_dict
        self.behaviors = behaviors
        self.neg_count = neg_count

    def __getitem__(self, idx):
        # generate positive and negative samples pairs under each behavior
        total = []

        pos = self.pos_sampling[idx]
        u_id = pos[0]

        all_inter = self.behavior_dict['all'].get(str(u_id), None)
        for index, behavior in enumerate(self.behaviors):
            tmp = []
            items = self.behavior_dict[behavior].get(str(u_id), None)
            if items is None:
                # [user, p_item, n_item*self.negcount, bhv_index, p_gt, n_gt*self.negcount] (self.neg_count + 1) is the ground truth label
                tmp = [0] * (self.neg_count + 3 + self.neg_count + 1)
            else:
                tmp.append(u_id)
                p_item = pos[1]
                tmp.append(p_item)
                for i in range(self.neg_count):
                    neg = random.randint(1, self.item_count)
                    while np.isin(neg, all_inter):
                        neg = random.randint(1, self.item_count)
                    tmp.append(neg)
                tmp.append(index)
                tmp.append(1)
                tmp.extend([0] * self.neg_count)
            total.append(tmp)

        return np.array(total)

    def __len__(self):
        return len(self.pos_sampling)


class BehaviorDate(Dataset):
    def __init__(self, user_count, item_count, pos_sampling, neg_count, behavior_dict=None, behaviors=None):
        self.user_count = user_count
        self.item_count = item_count
        self.pos_sampling = pos_sampling
        self.behavior_dict = behavior_dict
        self.behaviors = behaviors
        self.neg_count = neg_count

    def __getitem__(self, idx):
        # generate positive and negative samples pairs under each behavior
        total = []

        pos = self.pos_sampling[idx]
        u_id = pos[0]
        total.append(pos)

        all_inter = self.behavior_dict['all'].get(str(u_id), None)
        for i in range(self.neg_count):
            item = random.randint(1, self.item_count)
            while np.isin(item, all_inter):
                item = random.randint(1, self.item_count)
            neg = list(pos)
            neg[1] = item
            neg[-1] = 0
            total.append(neg)

        # random_items = np.setdiff1d(np.arange(1, self.item_count), list(all_inter))
        # random_items = np.random.choice(random_items, size=self.neg_count, replace=False)
        # for i in random_items:
        #     neg = pos.copy()
        #     neg[1] = i
        #     neg[-1] = 0
        #     total.append(neg)

        buy_inter = self.behavior_dict[self.behaviors[-1]].get(str(u_id), None)
        if buy_inter is None:
            signal = [0, 0, 0, 0]
        else:
            p_item = random.choice(buy_inter)
            n_item = random.randint(1, self.item_count)
            while np.isin(n_item, all_inter):
                n_item = random.randint(1, self.item_count)
            signal = [pos[0], p_item, n_item, 0]

        total.append(signal)

        return np.array(total)


    def __len__(self):
        return len(self.pos_sampling)


class DataSet(object):

    def __init__(self, args):

        self.train_data = None
        self.behaviors = args.behaviors
        self.path = args.data_path
        self.loss_type = args.loss_type
        self.neg_count = args.neg_count
        self.train_sample_path = args.train_sample_path

        self.__get_count()
        # self.__get_pos_sampling()
        self.__get_behavior_items()
        self.__get_validation_dict()
        self.__get_test_dict()
        self.__get_mask_dict()
        self.__get_sparse_interact_dict()

        self.validation_gt_length = np.array([len(x) for _, x in self.validation_interacts.items()])
        self.test_gt_length = np.array([len(x) for _, x in self.test_interacts.items()])

    def __get_count(self):
        with open(os.path.join(self.path, 'count.txt'), encoding='utf-8') as f:
            count = json.load(f)
            self.user_count = count['user']
            self.item_count = count['item']

    # def __get_pos_sampling(self):
    #     with open(os.path.join(self.path, 'pos_sampling.txt'), encoding='utf-8') as f:
    #         data = f.readlines()
    #         arr = []
    #         for line in data:
    #             line = line.strip('\n').strip().split()
    #             arr.append([int(x) for x in line])
    #         self.pos_sampling = arr

    def __get_behavior_items(self):
        """
        load the list of items corresponding to the user under each behavior
        :return:
        """
        self.train_behavior_dict = {}
        for behavior in self.behaviors:
            with open(os.path.join(self.path, behavior + '_dict.txt'), encoding='utf-8') as f:
                b_dict = json.load(f)
                self.train_behavior_dict[behavior] = b_dict
        with open(os.path.join(self.path, 'all_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.train_behavior_dict['all'] = b_dict

    # def __get_all_item_users(self):
    #     with open(os.path.join(self.path, 'all.txt'), encoding='utf-8') as f:

    def __get_test_dict(self):
        """
        load the list of items that the user has interacted with in the test set
        :return:
        """
        with open(os.path.join(self.path, 'test_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.test_interacts = b_dict

    def __get_validation_dict(self):
        """
        load the list of items that the user has interacted with in the validation set
        :return:
        """
        with open(os.path.join(self.path, 'validation_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.validation_interacts = b_dict

    def __get_mask_dict(self):

        valid_mask = defaultdict(list)
        for key, value in self.train_behavior_dict[self.behaviors[-1]].items():
            valid_mask[key].extend(value)
        for key, value in self.test_interacts.items():
            valid_mask[key].extend(value)
        self.valid_mask = dict(valid_mask)

        test_mask = defaultdict(list)
        for key, value in self.train_behavior_dict[self.behaviors[-1]].items():
            test_mask[key].extend(value)
        for key, value in self.validation_interacts.items():
            test_mask[key].extend(value)
        self.test_mask = dict(test_mask)

    def __get_sparse_interact_dict(self):
        """
        load graphs

        :return:
        """
        # self.edge_index = {}
        self.item_behaviour_degree = []
        self.user_behaviour_degree = []
        # self.all_item_user = {}
        # self.behavior_item_user = {}
        self.inter_matrix = []
        # self.user_item_inter_set = []
        all_row = []
        all_col = []
        for behavior in self.behaviors:
            # tmp_dict = {}
            with open(os.path.join(self.path, behavior + '.txt'), encoding='utf-8') as f:
                data = f.readlines()
                row = []
                col = []
                for line in data:
                    line = line.strip('\n').strip().split()
                    row.append(int(line[0]))
                    col.append(int(line[1]))

                    # if line[1] in self.all_item_user:
                    #     self.all_item_user[line[1]].append(int(line[0]))
                    # else:
                    #     self.all_item_user[line[1]] = [int(line[0])]

                    # if line[1] in tmp_dict:
                    #     tmp_dict[line[1]].append(int(line[0]))
                    # else:
                    #     tmp_dict[line[1]] = [int(line[0])]
                # self.behavior_item_user[behavior] = tmp_dict
                # indices = np.vstack((row, col))
                # indices = torch.LongTensor(indices)

                values = torch.ones(len(row), dtype=torch.float32)
                inter_matrix = sp.coo_matrix((values, (row, col)), [self.user_count + 1, self.item_count + 1])
                self.inter_matrix.append(inter_matrix)
                # user_item_set = [list(row.nonzero()[1]) for row in inter_matrix.tocsr()]
                # self.user_item_inter_set.append(user_item_set)
                user_degree = inter_matrix.sum(axis=1)
                item_degree = inter_matrix.sum(axis=0)
                user_degree = torch.from_numpy(user_degree).squeeze()
                item_degree = torch.from_numpy(item_degree).squeeze()
                self.item_behaviour_degree.append(item_degree)
                self.user_behaviour_degree.append(user_degree)

                # user_item_set = [set(row.nonzero()[1]) for row in user_item_inter_matrix]
                # item_user_set = [set(user_inter[:, j].nonzero()[0]) for j in range(user_inter.shape[1])]

                # user_inter = torch.sparse.FloatTensor(indices, values, [self.user_count + 1, self.item_count + 1]).to_dense()
                # self.item_behaviour_degree.append(user_inter.sum(dim=0))
                # self.user_behaviour_degree.append(user_inter.sum(dim=1))
                # col = [x + self.user_count + 1 for x in col]
                # row, col = [row, col], [col, row]
                # row = torch.LongTensor(row).view(-1)
                all_row.extend(row)
                # col = torch.LongTensor(col).view(-1)
                all_col.extend(col)
                # edge_index = torch.stack([row, col])
                # self.edge_index[behavior] = edge_index
        # self.all_item_user = {key: list(set(value)) for key, value in self.all_item_user.items()}
        self.item_behaviour_degree = torch.stack(self.item_behaviour_degree, dim=0).T
        self.user_behaviour_degree = torch.stack(self.user_behaviour_degree, dim=0).T
        # all_row = torch.cat(all_row, dim=-1)
        # all_col = torch.cat(all_col, dim=-1)
        # all_row = all_row.tolist()
        # all_col = all_col.tolist()
        # self.all_edge_index = list(set(zip(all_row, all_col)))
        all_edge_index = list(set(zip(all_row, all_col)))
        all_row = [sub[0] for sub in all_edge_index]
        all_col = [sub[1] for sub in all_edge_index]
        values = torch.ones(len(all_row), dtype=torch.float32)
        self.all_inter_matrix = sp.coo_matrix((values, (all_row, all_col)), [self.user_count + 1, self.item_count + 1])

        # self.all_edge_index = torch.LongTensor(self.all_edge_index).T


    # def behavior_dataset(self):
    #     return BehaviorDate(self.user_count, self.item_count, self.pos_sampling, self.neg_count, self.train_behavior_dict, self.behaviors)

    def train_dataset(self):
        # return TrainDate(self.train_data)
        return TrainDate(self.user_count, self.item_count, self.neg_count, self.train_behavior_dict, self.behaviors)
        # return TrainSample(self.user_count, self.item_count, self.pos_sampling, self.neg_count, self.train_behavior_dict, self.behaviors)

    def validate_dataset(self):
        return TestDate(self.user_count, self.item_count, samples=list(self.validation_interacts.keys()))

    def test_dataset(self):
        return TestDate(self.user_count, self.item_count, samples=list(self.test_interacts.keys()))

    # def load_train_samples(self, index):
    #     with open(os.path.join(self.path, self.train_sample_path, 'train_sampling_' + str(index % 100)) + '.txt', 'rb') as f:
    #         samples = json.load(f)
    #         self.train_data = samples

    # def generate_train_samples(self, path):
    #     # path = './samples'
    #     # if not os.path.exists(path):
    #     #     os.makedirs(path)
    #     all_inter = self.train_behavior_dict['all']
    #     for i in range(100):
    #         total = []
    #         with open(os.path.join(path, 'train_sampling_' + str(i) + '.txt'), 'w') as f:
    #             for index, behavior in enumerate(self.behaviors):
    #                 tmp_dict = self.train_behavior_dict[behavior]
    #                 for k in tmp_dict:
    #                     tmp_list = []
    #                     v = random.choice(tmp_dict[k])
    #                     pos = [int(k), v, index, 1]
    #                     tmp_list.append(pos)
    #
    #                     for i in range(self.neg_count):
    #                         item = random.randint(1, self.item_count)
    #                         while np.isin(item, all_inter):
    #                             item = random.randint(1, self.item_count)
    #                         neg = pos.copy()
    #                         neg[1] = item
    #                         neg[-1] = 0
    #                         tmp_list.append(neg)
    #
    #                     buy_inter = self.train_behavior_dict[self.behaviors[-1]].get(k, None)
    #                     if buy_inter is None:
    #                         signal = [0, 0, 0, 0]
    #                     else:
    #                         p_item = random.choice(buy_inter)
    #                         n_item = random.randint(1, self.item_count)
    #                         while np.isin(n_item, all_inter):
    #                             n_item = random.randint(1, self.item_count)
    #                         signal = [pos[0], p_item, n_item, 0]
    #                     tmp_list.append(signal)
    #                     total.append(tmp_list)
    #             f.write(json.dumps(total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set args', add_help=False)
    parser.add_argument('--behaviors', type=list, default=['click', 'cart', 'collect', 'buy'], help='')
    parser.add_argument('--data_path', type=str, default='../data/Tmall', help='')
    parser.add_argument('--loss_type', type=str, default='bpr', help='')
    parser.add_argument('--neg_count', type=int, default=4)
    parser.add_argument('--train_sample_path', type=str, default='samples')
    args = parser.parse_args()
    dataset = DataSet(args)
    # loader = DataLoader(dataset=dataset.behavior_dataset(), batch_size=5, shuffle=True)
    # for index, item in enumerate(loader):
    #     print(index, '-----', item)

    # path = os.path.join(args.data_path, args.train_sample_path)
    # if not os.path.exists(path):
    #     os.makedirs(path)
    #     dataset.generate_train_samples(path)

    # dataset.load_train_samples(0)


    loader = DataLoader(dataset=dataset.train_dataset(), batch_size=5, shuffle=False)
    for index, item in enumerate(loader):
        print(index, '-----', item)

