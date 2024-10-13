#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author:
# @Date  : 2024/8/23 16:16
# @Desc  :
import os.path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions.normal import Normal
from data_set import DataSet
from utils import BPRLoss, EmbLoss, distcorr
from lightGCN import LightGCN

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        return out


class BHV_PT(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(BHV_PT, self).__init__()
        self.device = args.device
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.layers = args.layers
        self.behaviors = args.behaviors
        self.behavior_Graphs = nn.ModuleList([
            LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, dataset.inter_matrix[index]) for
            index, _ in enumerate(self.behaviors)
        ])

    def gcn_propagate(self, embeddings):
        """
        gcn propagate in each behavior
        """
        all_embeddings = []
        for idx in range(len(self.behaviors)):
            behavior_embeddings = self.behavior_Graphs[idx](embeddings)
            all_embeddings.append(behavior_embeddings)
        return all_embeddings

    def forward(self, embeddings):
        all_embeddings = self.gcn_propagate(embeddings)
        return all_embeddings


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        self._gate_index = torch.nonzero(gates)
        sorted_experts, index_sorted_experts = self._gate_index.sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._rebuilt_batch = self._gate_index[index_sorted_experts[:, 1]]
        self._batch_index = self._rebuilt_batch[:, 0]
        self._rebuilt_index = self._batch_index * self._num_experts + self._rebuilt_batch[:, 1]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        # zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        zeros = torch.zeros(self._gates.size(0) * self._gates.size(1), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._rebuilt_index, stitched.float())
        combined = combined.view(self._gates.size(0), self._gates.size(1), expert_out[-1].size(1))

        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)





class MBLFE(nn.Module):
    def __init__(self, args, dataset: DataSet, noisy_gating=True):
        super(MBLFE, self).__init__()
        self.device = args.device
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.user_behaviour_degree = dataset.user_behaviour_degree
        self.item_behaviour_degree = dataset.item_behaviour_degree
        degrees = torch.cat([dataset.user_behaviour_degree, dataset.item_behaviour_degree], dim=0)
        self.behavior_weight = degrees / (degrees.sum(1, keepdim=True) + 1e-6)
        self.behavior_weight = self.behavior_weight.T.unsqueeze(-1).to(self.device)
        self.embedding_size = args.embedding_size
        self.behaviors = args.behaviors
        self.neg_count = args.neg_count
        self.irr_sample_no = args.irr_sample_no  # the number of irrelevant constraint
        self.label_size = args.label_size
        self.num_experts = args.num_experts
        self.noisy_gating = noisy_gating
        self.ssl_tau = args.ssl_tau
        self.batch_size = args.batch_size

        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        nn.init.xavier_uniform_(self.user_embedding.weight.data[1:])
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        nn.init.xavier_uniform_(self.item_embedding.weight.data[1:])

        self.bhv_pt = BHV_PT(args, dataset)
        self.label_experts = nn.ModuleList([MLP(self.embedding_size, self.label_size, self.embedding_size // 2) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(self.embedding_size, self.num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(self.embedding_size, self.num_experts), requires_grad=True)
        self.target_space = nn.Linear(self.label_size, self.label_size, bias=False)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))


        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model

        self.cross_loss = nn.BCEWithLogitsLoss()
        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()
        self.reg_weight = args.reg_weight
        self.ssl_reg = args.ssl_reg

        self.storage_user_embeddings = None
        self.storage_item_embeddings = None

        self.apply(self._init_weights)

        self._load_model()

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters, strict=False)

    def noisy_top_k_gating(self, x, train=True, noise_epsilon=1e-2):
        """Noisy top-k gating.
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        mean_weight = torch.mean(logits, dim=-1) - 1e-8
        gates = torch.where(logits >= mean_weight[:, None], logits, torch.zeros_like(logits))
        gates = gates / torch.sum(gates, dim=-1).unsqueeze(1)

        return gates

    def get_calculate_result(self, gates, embs, multiply_by_gates=True):
        '''
        get the export calculate result
        '''
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(embs)
        expert_outputs = [self.label_experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        result = dispatcher.combine(expert_outputs, multiply_by_gates)
        return result, expert_outputs

    def calculate_ssl_loss(self, user_expert_outputs, item_export_output):
        norm_user_export_output = [F.normalize(output, dim=1) for output in user_expert_outputs]
        norm_item_export_output = [F.normalize(output, dim=1) for output in item_export_output]

        def _calculate_ssl_loss(emb_list):
            loss = 0
            for idx in range(self.num_experts):
                tmp_tensor = emb_list[idx]
                tensor_size = tmp_tensor.size(0)
                if tensor_size < 2:
                    continue
                random_indices = random.sample(range(tensor_size), 2)
                a, b = tmp_tensor[random_indices]
                all_emb = [elem for i, elem in enumerate(emb_list) if i != idx]
                all_emb.append(b.unsqueeze(0))
                all_emb = torch.cat(all_emb, dim=0)
                v1 = torch.sum(a * b)
                v2 = a.matmul(all_emb.T)
                v1 = torch.exp(v1 / self.ssl_tau)
                v2 = torch.sum(torch.exp(v2 / self.ssl_tau))
                loss += -torch.log(v1 / v2)
            return loss / self.num_experts

        ssl_loss = 0
        epochs = self.batch_size // self.num_experts
        for _ in range(epochs):
            ssl_loss += _calculate_ssl_loss(norm_user_export_output)
            ssl_loss += _calculate_ssl_loss(norm_item_export_output)
        ssl_loss /= epochs
        return ssl_loss


    def forward(self, batch_data):

        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = self.bhv_pt(all_embeddings)
        # stage 1 training
        stage_1_loss = 0
        users = batch_data[:, 0, 0].long()
        for index, behavior in enumerate(self.behaviors):
            data = batch_data[:, index]
            items = data[:, 1:3].long()
            user_all_embedding, item_all_embedding = torch.split(all_embeddings[index], [self.n_users + 1, self.n_items + 1])
            user_feature = user_all_embedding[users.view(-1, 1)]
            item_feature = item_all_embedding[items]
            scores = torch.sum(user_feature * item_feature, dim=2)
            stage_1_loss += self.bpr_loss(scores[:, 0], scores[:, 1])
        stage_1_loss = stage_1_loss / len(self.behaviors)

        # stage 2 training label selecters
        agg_emb = torch.stack(all_embeddings, dim=0)
        agg_emb = torch.sum(agg_emb * self.behavior_weight, dim=0)
        user_agg_embs, item_agg_embs = torch.split(agg_emb, [self.n_users + 1, self.n_items + 1])
        items = batch_data[:, :, 1:6].long()
        b_user_embs = user_agg_embs[users]
        b_item_embs = item_agg_embs[items]
        b_ssl_item_embs = b_item_embs[:, -1, 0, :]
        b_item_embs_shape = b_item_embs.shape
        b_item_gate_input = b_item_embs.view(-1, b_item_embs_shape[-1])

        # user label match
        u_gates = self.noisy_top_k_gating(b_user_embs)

        user_export_embs, user_expert_outputs = self.get_calculate_result(u_gates, b_user_embs, multiply_by_gates=False)
        _, item_export_output = self.get_calculate_result(u_gates, b_ssl_item_embs, multiply_by_gates=False)

        ssl_loss = self.calculate_ssl_loss(user_expert_outputs, item_export_output)

        item_export_embs = [self.label_experts[i](b_item_gate_input) for i in range(self.num_experts)]
        item_export_embs = torch.cat(item_export_embs, dim=-1)
        item_export_embs = item_export_embs.view(b_item_embs_shape[0], b_item_embs_shape[1], b_item_embs_shape[2], -1)

        gt_values = batch_data[:, :, 7:]
        gt_values = gt_values.reshape(-1, 1)
        scores = torch.sum(user_export_embs.view(user_export_embs.shape[0], 1, 1, -1) * item_export_embs, dim=-1)
        scores = scores.view(-1, 1)
        log_loss = self.cross_loss(scores, gt_values.float())

        stage_2_loss = log_loss + self.ssl_reg * ssl_loss

        # stage 3 training target behavior recommendation
        target_space_user_embs = self.target_space(user_export_embs).view(user_export_embs.shape[0], 1, -1)
        target_item_pairs = item_export_embs[:, -1, 0:2, :]
        scores = torch.sum(target_space_user_embs * target_item_pairs, dim=2)
        stage_3_loss = self.bpr_loss(scores[:, 0], scores[:, 1])

        total_loss = stage_1_loss + stage_2_loss + stage_3_loss + self.reg_weight * self.emb_loss(self.user_embedding.weight, self.item_embedding.weight)

        return total_loss

    def full_predict(self, users):
        if self.storage_user_embeddings is None:
            all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
            all_embeddings = self.bhv_pt(all_embeddings)
            agg_emb = torch.stack(all_embeddings, dim=0)
            agg_emb = torch.sum(agg_emb * self.behavior_weight, dim=0)
            user_agg_embs, item_agg_embs = torch.split(agg_emb, [self.n_users + 1, self.n_items + 1])

            ids = torch.arange(0, self.n_users + 1)
            label_users = []
            for i in range(0, self.n_users + 1, 1024):
                tmp_ids = ids[i: i + 1024]
                u_emb = user_agg_embs[tmp_ids]
                u_gates = self.noisy_top_k_gating(u_emb, train=False)
                user_export_embs, _ = self.get_calculate_result(u_gates, u_emb, multiply_by_gates=False)
                target_space_user_embs = self.target_space(user_export_embs).view(user_export_embs.shape[0], -1)
                label_users.append(target_space_user_embs)
            self.storage_user_embeddings = torch.cat(label_users, dim=0)

            ids = torch.arange(0, self.n_items + 1)
            label_items = []
            for i in range(0, self.n_items + 1, 1024):
                tmp_ids = ids[i: i + 1024]
                i_emb = item_agg_embs[tmp_ids]
                item_export_embs = [self.label_experts[i](i_emb) for i in range(self.num_experts)]
                item_export_embs = torch.cat(item_export_embs, dim=-1)
                item_export_embs = item_export_embs.view(item_export_embs.shape[0], -1)
                label_items.append(item_export_embs)
            self.storage_item_embeddings = torch.cat(label_items, dim=0)
            self.storage_item_embeddings = self.storage_item_embeddings.transpose(0, 1)

        user_emb = self.storage_user_embeddings[users.long()]
        scores = torch.matmul(user_emb, self.storage_item_embeddings)

        return scores

