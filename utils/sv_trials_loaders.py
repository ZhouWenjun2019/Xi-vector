#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:15:48 2020

@author: shreyasr
"""

import re
import numpy as np
import random
import sys
import subprocess
import pickle
import os
import torch
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, Subset
sys.path.insert(0, '/home/zhouwj/kaldi/egs/zhouwj/sre/subtools/pytorch/libs/support')
import kaldi_io


class TrialSampler:
    def __init__(self, spk2utt_file, 
                 batch_size, num_spks_per_batch, 
                 mega_scp_dict, id_to_num_dict):
        with open(spk2utt_file) as f:
            self.spk2utt_list = f.readlines()
        random.shuffle(self.spk2utt_list)
        self.batch_size = batch_size
        self.num_spks_per_batch = num_spks_per_batch
        self.num_utts_per_spk = batch_size / num_spks_per_batch
        self.n_repeats_tgt = int(0.7*(self.num_utts_per_spk - 1))
        self.n_repeats_imp = int(0.7 * self.num_utts_per_spk * (self.num_spks_per_batch - 1))
        self.mega_scp_dict = mega_scp_dict
        self.id_to_num_dict = id_to_num_dict
        self.spk2utt_dict = {}
        
    def spk2utt_dict_from_list(self):
        for x in self.spk2utt_list:
            a = np.asarray(x.rstrip('\n').split(' ', 1)[1].split(' '))
            random.shuffle(a)
            self.spk2utt_dict[x.split(' ', 1)[0]] = np.array_split(a, np.ceil(a.shape[0] / self.num_utts_per_spk))
            
    def check_spk2utt_dict(self):
        keys_to_remove=[]
        for k, v in self.spk2utt_dict.items():
            if len(self.spk2utt_dict[k]) == 0:
                keys_to_remove.append(k)
        if len(keys_to_remove) > 0:
            for k in keys_to_remove:
                del self.spk2utt_dict[k]
        return len(self.spk2utt_dict.keys()) > self.num_spks_per_batch
    
    def get_batch(self):
        spk2utt_keys = list(self.spk2utt_dict.keys())
        random.shuffle(spk2utt_keys)
        keys_to_sample = spk2utt_keys[:self.num_spks_per_batch]
        sampled_keys_utts_per_spk = []
        diff_speaker_spk2utt_dict = {}
        for x in keys_to_sample:
            sampled_keys_utts_per_spk.append(self.spk2utt_dict[x][0])
            diff_speaker_spk2utt_dict[x] = self.spk2utt_dict[x][0]
            del self.spk2utt_dict[x][0]
        t1, t2 = same_speaker_list(sampled_keys_utts_per_spk, 
                                   self.mega_scp_dict, 
                                   self.id_to_num_dict, 
                                   n_repeats=self.n_repeats_tgt
        )
        nt1, nt2 = diff_speaker_list(diff_speaker_spk2utt_dict, 
                                     self.mega_scp_dict, 
                                     self.id_to_num_dict, 
                                     n_repeats=self.n_repeats_imp
        )
        targets = torch.ones(t1.size())
        non_targets = torch.zeros(nt1.size())
        d1, d2, labels = torch.cat((t1,nt1)).float(), torch.cat((t2,nt2)).float(), torch.cat((targets,non_targets)).float()
        return d1, d2, labels
        
    def load_epoch(self):
        self.spk2utt_dict_from_list()
        epoch_data = []
        while self.check_spk2utt_dict():
            epoch_data.append(self.get_batch())
        return epoch_data


def custom_loader_e2e(data_spk2utt_list, mega_scp_dict, 
                      id_to_num_dict, batch_size=64, 
                      num_spks_per_batch=4):
    mega_loader = []
    if type(data_spk2utt_list) == str:
        data_spk2utt_list = np.genfromtxt(data_spk2utt_list, dtype='str')
    if data_spk2utt_list.ndim == 2:
        data_spk2utt_list = data_spk2utt_list[:, 0]
    else:
        raise("Something wrong here.")
    for spk2utt_file in data_spk2utt_list:
        ts = TrialSampler(spk2utt_file, batch_size, 
                          num_spks_per_batch, 
                          mega_scp_dict, 
                          id_to_num_dict
        )
        mega_loader.extend(ts.load_epoch())
    random.shuffle(mega_loader)
    return mega_loader

def custom_loader_e2e_v2(nc, mega_scp_dict, id_to_num_dict):
    mega_loader = []
    for spk2utt_file in nc.train_spk2utt_list:
        for num_spks_per_batch in range(nc.min_num_spks_per_batch, nc.max_num_spks_per_batch+1):
            ts = TrialSampler(spk2utt_file, nc.batch_size, 
                              num_spks_per_batch, 
                              mega_scp_dict, 
                              id_to_num_dict
            )
            mega_loader.extend(ts.load_epoch())
    random.shuffle(mega_loader)
    return mega_loader
       
def same_speaker_list(utts_per_spk, combined_scp_dict, 
                      id_to_num_dict, n_repeats=1):
    d1, d2 = [], []
    for repeats in range(n_repeats):
        for utts in utts_per_spk:
            utts_shuffled = list(utts.copy())
            random.shuffle(utts_shuffled)
            while len(utts_shuffled) >= 2:
                tmp1 = utts_shuffled.pop()
                if tmp1 not in combined_scp_dict:
                    continue
                tmp2 = utts_shuffled.pop()
                if tmp2 not in combined_scp_dict:
                    continue
                d1.append(id_to_num_dict[tmp1])
                d2.append(id_to_num_dict[tmp2])
    d1, d2 = torch.tensor(d1), torch.tensor(d2)
    return d1, d2

def diff_speaker_list(spk2utt_dict, combined_scp_dict, 
                      id_to_num_dict, n_repeats=1):
    spk2utt_keys = list(spk2utt_dict.keys())
    utt2spk = []
    for i in spk2utt_keys:
        for j in spk2utt_dict[i]:
            utt2spk.append([j, i])
    d1, d2 = [], []
    for repeats in range(n_repeats):
        utt2spk_list = list(utt2spk)
        random.shuffle(utt2spk_list)
        i = 0
        while len(utt2spk_list) >= 2:
            if utt2spk_list[-1][1] != utt2spk_list[-2][1]:
                tmp1 = utt2spk_list.pop()
                if list(tmp1)[0] not in combined_scp_dict:
                    continue
                tmp2 = utt2spk_list.pop()
                if list(tmp2)[0] not in combined_scp_dict:
                    continue
                d1.append(id_to_num_dict[list(tmp1)[0]])
                d2.append(id_to_num_dict[list(tmp2)[0]])
                i = 0
            else:
                i = i + 1
                random.shuffle(utt2spk_list)
                if i == 50:
                    break
    d1, d2 = torch.tensor(d1), torch.tensor(d2)
    return d1, d2

def make_same_speaker_list(spk2utt_file, 
                           combined_scp_dict, 
                           n_repeats=1, 
                           train_and_valid=False,
                           train_ratio=0.95):
    '''
    获取正样本列表: <uttID uttID>
    spk2utt_file: spk2utt文件列表
    combined_scp_dict: 需要包含spk2utt中所有utts
    n_repeats: 数据重复使用次数
    train_and_valid: 若为true，返回{train,valid}列表；否则，返回一个总列表
    train_ratio: 训练模型时，训练与验证数据的划分比
    '''
    print("In same speaker list...")
    assert train_ratio < 1, "train_ratio should be less than 1."
    with open(spk2utt_file) as f:
        spk2utt_list = f.readlines()
    random.seed(2)
    random.shuffle(spk2utt_list)
    uttsperspk = [(a.rstrip('\n').split(' ', 1)[1]).split(' ') for a in spk2utt_list] # 记录每个spk的utts
    
    train_uttsperspk = uttsperspk[:int(train_ratio * len(uttsperspk))]
    valid_uttsperspk = uttsperspk[int((train_ratio) * len(uttsperspk)):]

    train_same_speaker_list = []
    for repeats in range(n_repeats):
        for utts in train_uttsperspk: # 一个spk的所有utts
            utts_shuffled = utts.copy() # 浅拷贝
            random.shuffle(utts_shuffled)
            while len(utts_shuffled) >= 2:
                tmp1 = utts_shuffled.pop()
                if tmp1 not in combined_scp_dict:
                    print(tmp1 +'did not in the dict')
                    continue
                tmp2 = utts_shuffled.pop()
                if tmp2 not in combined_scp_dict:
                    print(tmp2 +'did not in the dict')
                    continue
                train_same_speaker_list.append([tmp1, tmp2]) # 不会有重叠
    train_same_speaker_list = np.asarray(train_same_speaker_list)
    
    valid_same_speaker_list = []
    for repeats in range(n_repeats):
        for utts in valid_uttsperspk:
            utts_shuffled = utts.copy()
            random.shuffle(utts_shuffled)
            while len(utts_shuffled) >= 2:
                tmp1 = utts_shuffled.pop()
                if tmp1 not in combined_scp_dict:
                    print(tmp1 +'did not in the dict')
                    continue
                tmp2 = utts_shuffled.pop()
                if tmp2 not in combined_scp_dict:
                    print(tmp2 +'did not in the dict')
                    continue
                valid_same_speaker_list.append([tmp1, tmp2])
    valid_same_speaker_list = np.asarray(valid_same_speaker_list)

    if train_and_valid:
        return train_same_speaker_list, valid_same_speaker_list
    else:
        return train_same_speaker_list + valid_same_speaker_list

def make_diff_speaker_list(spk2utt_file, combined_scp_dict, 
                           n_repeats=1, train_and_valid=True, 
                           train_ratio=0.95):
    '''
    获取负样本列表: <uttID uttID>
    spk2utt_file: spk2utt文件列表
    combined_scp_dict: 需要包含spk2utt中所有utts
    n_repeats: 数据重复使用次数
    train_and_valid: 若为true，返回{train,valid}列表；否则，返回一个总列表
    train_ratio: 训练模型时，训练与验证数据的划分比
    '''
    print("In diff speaker list...")
    assert train_ratio < 1, "train_ratio should be less than 1."
    with open(spk2utt_file) as f:
        spk2utt_list = f.readlines()
    random.seed(2)
    random.shuffle(spk2utt_list)
    spk2utt_dict = {x.split(' ', 1)[0]: x.rstrip('\n').split(' ', 1)[1].split(' ') for x in spk2utt_list}

    spk2utt_keys = list(spk2utt_dict.keys())
    train_keys = spk2utt_keys[:int(train_ratio * len(spk2utt_keys))]
    valid_keys = spk2utt_keys[int(train_ratio * len(spk2utt_keys)):]

    utt2spk_train = []
    utt2spk_valid = []
    for i in train_keys:
        for j in spk2utt_dict[i]:
            utt2spk_train.append([j, i])
    for i in valid_keys:
        for j in spk2utt_dict[i]:
            utt2spk_valid.append([j, i])

    train_diff_speaker_list = []
    for repeats in range(n_repeats):
        utt2spk_list = list(utt2spk_train)
        random.shuffle(utt2spk_list)
        i = 0 
        while len(utt2spk_list) >= 2:
            if utt2spk_list[-1][1] != utt2spk_list[-2][1]: # 最后两个说话人不同
                tmp1 = utt2spk_list.pop()
                if list(tmp1)[0] not in combined_scp_dict:
                    print(list(tmp1)[0] +'did not in the dict')
                    continue
                tmp2 = utt2spk_list.pop()
                if list(tmp2)[0] not in combined_scp_dict:
                    print(list(tmp2)[0] +'did not in the dict')
                    continue
                train_diff_speaker_list.append([list(tmp1)[0], list(tmp2)[0]])
                i = 0
            else:
                i = i + 1
                random.shuffle(utt2spk_list)
                if i == 50: # 连续50utts的spk一致
                    break

    valid_diff_speaker_list = []
    for repeats in range(n_repeats):
        utt2spk_list = list(utt2spk_valid)
        random.shuffle(utt2spk_list)
        i = 0
        while len(utt2spk_list) >= 2:
            if utt2spk_list[-1][1] != utt2spk_list[-2][1]:
                tmp1 = utt2spk_list.pop()
                if list(tmp1)[0] not in combined_scp_dict:
                    continue
                tmp2 = utt2spk_list.pop()
                if list(tmp2)[0] not in combined_scp_dict:
                    continue
                valid_diff_speaker_list.append([list(tmp1)[0], list(tmp2)[0]])
                i = 0
            else:
                i = i + 1
                random.shuffle(utt2spk_list)
                if i == 50:
                    break
    
    train_diff_speaker_list = np.asarray(train_diff_speaker_list)
    valid_diff_speaker_list = np.asarray(valid_diff_speaker_list)
    
    if train_and_valid:
        return train_diff_speaker_list, valid_diff_speaker_list
    else:
        return train_diff_speaker_list + valid_diff_speaker_list
    
def generate_train_trial_keys(data_spk2utt_list, 
                              xvector_scp_list, 
                              train_and_valid=True, 
                              train_ratio=0.95):
    '''
    按照比例生成<utt utt label>的trials列表
    data_spk2utt_list: 
    xvector_scp_list: 所有数据的xVector地址
    train_and_valid: 是否划分{train,valid}
    train_ratio: 划分比
    '''
    
    xvector_scp_combined = {}
    
    for fx in xvector_scp_list:
        with open(fx) as f:
            scp_list = f.readlines()
        # 生成utt对应xvector的字典
        scp_dict = {os.path.splitext(os.path.basename(x.split(' ', 1)[0]))[0]: x.rstrip('\n').split(' ', 1)[1] for x in scp_list}
        xvector_scp_combined.update(scp_dict)

    if type(data_spk2utt_list) == str: # not here
        data_spk2utt_list = np.genfromtxt(data_spk2utt_list, dtype='str')

    if data_spk2utt_list.ndim == 2: 
        num_repeats_list = data_spk2utt_list[:, 1].astype(int)
        data_spk2utt_list = data_spk2utt_list[:, 0]
    elif data_spk2utt_list.ndim == 1:
        num_repeats_list = np.ones(len(data_spk2utt_list)).astype(int)
    else:
        raise("Something wrong here.")


    sampled_list_train = []
    sampled_list_valid = []

    for i, d in enumerate(data_spk2utt_list):
        print("In for loop get train dataset")
        same_train_list, same_valid_list = make_same_speaker_list(
                    d, xvector_scp_combined, 
                    # xvector_scp_list, 
                    n_repeats=num_repeats_list[i], 
                    train_and_valid=True, 
                    train_ratio=train_ratio
        )

        diff_train_list, diff_valid_list = make_diff_speaker_list(
                    d, xvector_scp_combined, 
                    n_repeats=num_repeats_list[i], 
                    train_and_valid=True, 
                    train_ratio=train_ratio
        )

        print(str(len(same_train_list)) +'\t'+ 
              str(len(same_valid_list)) +'\t'+ 
              str(len(diff_train_list)) +'\t'+ 
              str(len(diff_valid_list))
        )

        zeros = np.zeros((diff_train_list.shape[0], 1)).astype(int)
        ones = np.ones((same_train_list.shape[0], 1)).astype(int)
        same_list_with_label_train = np.concatenate((same_train_list, ones), axis=1)
        diff_list_with_label_train = np.concatenate((diff_train_list, zeros), axis=1)

        zeros = np.zeros((diff_valid_list.shape[0], 1)).astype(int)
        ones = np.ones((same_valid_list.shape[0], 1)).astype(int)
        same_list_with_label_valid = np.concatenate((same_valid_list, ones), axis=1)
        diff_list_with_label_valid = np.concatenate((diff_valid_list, zeros), axis=1)

        concat_pair_list_train = np.concatenate((
                            same_list_with_label_train, 
                            diff_list_with_label_train
        ))
        concat_pair_list_valid = np.concatenate((
                            same_list_with_label_valid, 
                            diff_list_with_label_valid
        ))

        np.random.shuffle(concat_pair_list_train)
        sampled_list_train.extend(concat_pair_list_train)

        np.random.shuffle(concat_pair_list_valid)
        sampled_list_valid.extend(concat_pair_list_valid)
    
    if train_and_valid:
        return sampled_list_train, sampled_list_valid
    else:
        return sampled_list_train + sampled_list_valid
    
def save_unique_train_valid_xvector_scps(data_spk2utt_list, 
                                         xvector_scp_list, 
                                         train_scp_path, 
                                         valid_scp_path, 
                                         train_ratio=0.95):
    '''
    data_spk2utt_list:
    xvector_scp_list:
    train_scp_path:
    valid_scp_path:
    train_ratio:
    '''
    if type(data_spk2utt_list) == str:
        data_spk2utt_list = np.genfromtxt(data_spk2utt_list, dtype='str')

    if data_spk2utt_list.ndim == 2:
        data_spk2utt_list = data_spk2utt_list[:, 0]
        
    xvector_scp_combined = {}
    
    for fx in xvector_scp_list:
        with open(fx) as f:
            scp_list = f.readlines()
        scp_dict = {x.split(' ', 1)[0]: x.rstrip('\n').split(' ', 1)[1] for x in scp_list}
        xvector_scp_combined.update(scp_dict)
    
    train_scp = []
    valid_scp = []

    for i, d in enumerate(data_spk2utt_list):
        with open(d) as f:
            spk2utt_list = f.readlines()
        random.seed(2)
        random.shuffle(spk2utt_list)
        spk2utt_dict = {x.split(' ', 1)[0]: x.rstrip('\n').split(' ', 1)[1].split(' ') for x in spk2utt_list}
        spks = list(spk2utt_dict.keys())
        train_keys = spks[:int(train_ratio * len(spks))]
        valid_keys = spks[int(train_ratio * len(spks)):]

        for i in train_keys:
            for j in spk2utt_dict[i]:
                if j in xvector_scp_combined:
                    train_scp.append([j, xvector_scp_combined[j]])
        for i in valid_keys:
            for j in spk2utt_dict[i]:
                if j in xvector_scp_combined:
                    valid_scp.append([j, xvector_scp_combined[j]])
    train_scp = np.asarray(train_scp)
    valid_scp = np.asarray(valid_scp)
    subprocess.call(['mkdir', '-p', os.path.dirname(train_scp_path)])
    subprocess.call(['mkdir', '-p', os.path.dirname(valid_scp_path)])
    np.savetxt(train_scp_path, train_scp, fmt='%s', delimiter=' ', comments='')
    np.savetxt(valid_scp_path, valid_scp, fmt='%s', delimiter=' ', comments='')

def combine_trials_and_get_loader(trials_key_files_list, 
                                  id_to_num_dict, 
                                  subsample_factors=None, 
                                  batch_size=2048, subset=0):
    if subsample_factors is None:
        subsample_factors = [1 for w in trials_key_files_list]
    datasets = []
    for f, sf in zip(trials_key_files_list, subsample_factors):
        t = np.genfromtxt(f, dtype = 'str')
        x1, x2, l = [], [], []
        for tr in t:
            try:
                a, b, c = id_to_num_dict[tr[0]], id_to_num_dict[tr[1]], float(tr[2])
                x1.append(a); x2.append(b); l.append(c)
            except:
                pass
        tdset = TensorDataset(torch.tensor(x1),torch.tensor(x2),torch.tensor(l))
        inds = np.arange(len(tdset))[np.random.rand(len(tdset))<sf]
        datasets.append(Subset(tdset, inds))
    combined_dataset = ConcatDataset(datasets)
    if subset > 0:
        inds = np.arange(len(combined_dataset))[np.random.rand(len(combined_dataset))<subset]
        combined_dataset = Subset(combined_dataset, inds)
    trials_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    return trials_loader

def get_source_loader(trials_key_files_list, 
                      id_to_num_dict, 
                      subsample_factors=None, 
                      batch_size=2048):
    '''
    合并trials文件为一个loader，并分批
    trials_key_files_list: <utt utt label>格式的训练trials
    id_to_num_dict: trials_key_files_list中涉及的spkID及其序号
    '''
    if subsample_factors is None:
        subsample_factors = [1 for w in trials_key_files_list]
    
    datasets = []
    for f, sf in zip(trials_key_files_list, subsample_factors):
        t = np.genfromtxt(f, dtype='str')
        x1, x2, l = [], [], [] # 拆分trials
        for tr in t:
            try:
                x1.append(id_to_num_dict[tr[0]])
                x2.append(id_to_num_dict[tr[1]])
                l.append(float(tr[2]))
            except:
                pass
        
        tdset = TensorDataset(torch.tensor(x1),
                              torch.tensor(x2), 
                              torch.tensor(l)
        )
        
        # 随机抽样
        inds = np.arange(len(tdset))[np.random.rand(len(tdset))<sf]
        datasets.append(Subset(tdset, inds))

    combined_dataset = ConcatDataset(datasets)
    
    trials_loader = DataLoader(combined_dataset, 
                               batch_size=batch_size, 
                               shuffle=True, 
                            #    shuffle=False
    )
    return trials_loader

def get_target_loader(trials_key_files_list, 
                      id_to_num_dict, 
                      subsample_factors=None, 
                      batch_size=2048):
    '''
    合并trials文件为一个loader，并分批
    trials_key_files_list: <utt utt label>格式的训练trials
    id_to_num_dict: trials_key_files_list中涉及的spkID及其序号
    '''
    if subsample_factors is None:
        subsample_factors = [1 for w in trials_key_files_list]
    
    datasets = []
    for f, sf in zip(trials_key_files_list, subsample_factors):
        t = np.genfromtxt(f, dtype='str')
        x1, x2, l = [], [], [] # 拆分trials
        for tr in t:
            try:
                x1.append(id_to_num_dict[tr[0]])
                x2.append(id_to_num_dict[tr[1]])
                l.append(float(tr[2]))
            except:
                print("Something wrong")
                pass
        
        tdset = TensorDataset(torch.tensor(x1),
                              torch.tensor(x2), 
                              torch.tensor(l)
        )
        
        # 随机抽样，应该不成问题，因为会都选上
        inds = np.arange(len(tdset))[np.random.rand(len(tdset))<sf]
        datasets.append(Subset(tdset, inds))

    combined_dataset = ConcatDataset(datasets)
    
    trials_loader = DataLoader(combined_dataset, 
                               batch_size=batch_size, 
                               shuffle=True, 
                            #    shuffle=False
    )
    return trials_loader

def get_trials_loaders_dict(trials_key_files_list, 
                            id_to_num_dict, subsample_factors=None, 
                            batch_size=2048, subset=0):
    '''
    为每个trials文件准备一个loader，并分批
    '''
    # 每个trials文件有一个loader
    trials_loaders_dict = {} 
    if subsample_factors is None:
        subsample_factors = [1 for w in trials_key_files_list]

    for f, sf in zip(trials_key_files_list, subsample_factors):
        t = np.genfromtxt(f, dtype = 'str')
        x1, x2, l = [], [], []
        for tr in t:
            try:
                x1.append(id_to_num_dict[tr[0]])
                x2.append(id_to_num_dict[os.path.splitext(tr[1])[0]])
                l.append(float(tr[2]))
            except:
                pass
        
        tdset = TensorDataset(torch.tensor(x1),torch.tensor(x2),torch.tensor(l))
        inds = np.arange(len(tdset))[np.random.rand(len(tdset))<sf]
        dataset = Subset(tdset, inds)

        if subset > 0:
            inds = np.arange(len(dataset))[np.random.rand(len(dataset))<subset]
            dataset = Subset(dataset, inds)
        
        trials_loader = DataLoader(dataset, 
                                   batch_size=batch_size, 
                                   shuffle=True
        )
        trials_loaders_dict[os.path.splitext(os.path.basename(f))[0]] = trials_loader
    return trials_loaders_dict

def load_xvec_trials_from_numbatch(mega_dict, 
                                   num_to_id_dict, 
                                   data1, data2, device):
    '''
    根据uttID加载xvec
    '''
    data1_xvec, data2_xvec = [], []
    for i, (d1, d2) in enumerate(zip(data1, data2)):
        data1_xvec_temp = mega_dict[num_to_id_dict[int(d1)]]
        data2_xvec_temp = mega_dict[num_to_id_dict[int(d2)]]
        data1_xvec.append(data1_xvec_temp)
        data2_xvec.append(data2_xvec_temp)
    tensor_X1 = torch.from_numpy(np.asarray(data1_xvec)).float().to(device)
    tensor_X2 = torch.from_numpy(np.asarray(data2_xvec)).float().to(device)
    return tensor_X1, tensor_X2






    