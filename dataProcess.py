'''
TIMIT用于训练NPLDA模型，Aishell用于验证Deep CORAL的有效性
TIMIT中说话人目录的首字母即性别标识：男性438人，女性192人
为保持性别平衡，男女各使用192人，300人用于训练，84人用于测试
'''

import glob
import shutil
from tqdm import tqdm
import os
import numpy as np
import pickle
import sys
from utils.sv_trials_loaders import generate_train_trial_keys
import kaldi_io
import subprocess

def generate_source_data():
    '''
    从数据库中提取实验需要的源域wav文件，确保性别比均衡
    '''
    timitRoot = '/mnt/h/Database/English/TIMIT/wav'
    dstRoot = '/mnt/d/Database/timit/wav'

    spkListMale = []
    spkListFemale = []
    wavList = sorted(glob.glob(timitRoot +'/*/*.wav'))
    for wav in wavList:
        spk = wav.split('/')[-2]
        if (spk[0] == 'M') and (spk not in spkListMale):
            spkListMale.append(spk)
        elif (spk[0] == 'F') and (spk not in spkListFemale):
            spkListFemale.append(spk)

    nSelect = min(len(spkListMale), len(spkListFemale))

    # TODO:修改为随机采样
    ratio = 0.8 # 训练集占比
    subMaleTrain = spkListMale[:int(ratio*nSelect)]
    subMaleTest = spkListMale[int(ratio*nSelect) : nSelect]
    subFemaleTrain = spkListFemale[:int(ratio*nSelect)]
    subFemaleTest = spkListFemale[int(ratio*nSelect) : nSelect]
    subTrain = subMaleTrain + subFemaleTrain
    subTest = subMaleTest + subFemaleTest

    for spk in tqdm(subTrain):
        srcdir = timitRoot +'/'+ spk
        dstdir = dstRoot +'/train/'+ spk
        shutil.copytree(srcdir, dstdir)

    for spk in tqdm(subTest):
        srcdir = timitRoot +'/'+ spk
        dstdir = dstRoot +'/test/'+ spk
        shutil.copytree(srcdir, dstdir)
    
    print("Source data has been prepared !")

def generate_target_data(ratio=0.8, nTrain=10, nTest=10, nSelect=0, 
                         dstRoot = '/mnt/d/Database/aishell/wav'):
    '''
    从数据库中提取实验需要的目标域wav文件，确保性别比均衡
    '''
    srcRoot = '/mnt/h/Database/Chinese/aishell/wav'
    spkInfo = '/mnt/h/Database/Chinese/aishell/speaker.info'

    # 得到性别字典
    spk_gender_dict = {}
    with open(spkInfo, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            spk = line.strip('\n').split(' ')[0]
            gender = line.strip('\n').split(' ')[1]
            spk_gender_dict[spk] = gender

    spkListMale = []
    spkListFemale = []
    wavList = sorted(glob.glob(srcRoot +'/*/*.wav'))
    for wav in wavList:
        spk = wav.split('/')[-2]
        if (spk_gender_dict[spk[1:]] == 'M') and (spk not in spkListMale):
            spkListMale.append(spk)
        elif (spk_gender_dict[spk[1:]] == 'F') and (spk not in spkListFemale):
            spkListFemale.append(spk)

    if nSelect == 0:
        nSelect = min(len(spkListMale), len(spkListFemale)) # 确保男女均衡

    # TODO:修改为随机采样
    subMaleTrain = spkListMale[:int(ratio*nSelect)]
    subMaleTest = spkListMale[int(ratio*nSelect) : nSelect]
    subFemaleTrain = spkListFemale[:int(ratio*nSelect)]
    subFemaleTest = spkListFemale[int(ratio*nSelect) : nSelect]
    subTrain = subMaleTrain + subFemaleTrain
    subTest = subMaleTest + subFemaleTest

    for spk in tqdm(subTrain):
        srcdir = srcRoot +'/'+ spk
        dstdir = dstRoot +'/train/'+ spk
        if not os.path.exists(dstdir):
            os.makedirs(dstdir)
        wavList = sorted(glob.glob(srcdir +'/*.wav'))[:nTrain]
        for wav in wavList:
            shutil.copy(wav, dstdir)

    for spk in tqdm(subTest):
        srcdir = srcRoot +'/'+ spk
        dstdir = dstRoot +'/test/'+ spk
        if not os.path.exists(dstdir):
            os.makedirs(dstdir)

        wavList = sorted(glob.glob(srcdir +'/*.wav'))[:nTest]
        for wav in wavList:
            shutil.copy(wav, dstdir)

    print("Target data has been prepared !")

def split_enroll_test():
    '''
    划分注册集和验证集
    '''
    os.chdir('/home/zhouwj/kaldi/egs/zhouwj/sre')
    os.system('./dataProcessDomain.sh')
    os.chdir('/home/zhouwj/code/domainAdapt')

def split_train_valid():
    '''
    划分训练网络时的训练集和验证集
    Based on: Neural PLDA--dataprep_domain.py
    '''
    data_spk2utt_list = np.asarray([
    ['./data/aishell_train/spk2utt', '2']
    ])
    xvector_scp_list = np.asarray([
        './exp/xvectors_aishell_train/xvectorPro.scp'
    ])

    # 这里的val_trial_keys是训练集的子集，验证集还需要单独划分
    train_trial_keys, val_trial_keys = generate_train_trial_keys(
                                        data_spk2utt_list, 
                                        xvector_scp_list, 
                                        train_and_valid=True, 
                                        train_ratio=0.95
    )

    np.savetxt('trials_and_keys/aishell_train.tsv', 
               train_trial_keys, fmt='%s', 
               delimiter='\t', comments='none'
    )
    np.savetxt('trials_and_keys/aishell_valid.tsv', 
               val_trial_keys, fmt='%s', 
               delimiter='\t', comments='none'
    )

    print("Train set and valid set have done !")

def get_mega_xvectors(xvector_scp_list=[], 
                      save_flag=False, 
                      dstPath='exp/mega_xvector.pkl'):
    '''
    加载/保存所有的特征文件
    '''
    
    if xvector_scp_list == []:
        xvector_scp_list = np.asarray([
            './exp/xvectors_timit_train/xvector.scp', # 3060
            './exp/xvectors_timit_enroll/spk_xvector.scp', # 78
            './exp/xvectors_timit_eval/xvector.scp', # 156
            './exp/xvectors_aishell_train/xvector.scp', # 2960
            './exp/xvectors_aishell_enroll/xvector.scp', # 608
            './exp/xvectors_aishell_enroll/spk_xvector.scp', # 76
            './exp/xvectors_aishell_eval/xvector.scp', # 152
        ])
    
    mega_xvec_dict = {}
    for fx in xvector_scp_list:
        # # 改路径
        # subprocess.call(['sed', '-i', 's| {}| {}|g'.format('/home/zhouwj/domain', '/home/zhouwj/code'), fx])
        with open(fx) as f:
            scp_list = f.readlines()

        # scp_dict = {}
        xvec_dict = {}
        for x in scp_list: 
            key = os.path.splitext(os.path.basename(x.replace('\t', ' ').split(' ', 1)[0]))[0] # uttID
            # value1 = x.rstrip('\n').split(' ', 1)[1] # fea address
            # scp_dict[key] = value1
            value2 = kaldi_io.read_vec_flt(x.rstrip('\n').replace('\t', ' ').split(' ', 1)[1])
            xvec_dict[key] = value2
        mega_xvec_dict.update(xvec_dict) 


    if save_flag:
        with open(dstPath, 'w') as fa:
            pickle.dump(mega_xvec_dict, fa)
        print("All xvectors has been saved !")
    else:
        return mega_xvec_dict


def make_domain_label_list():
    '''
    格式：<uttID> <domain Label>
    '''
    timit_scp_list = [
        './exp/xvectors_timit_train/xvector.scp',
        # './exp/xvectors_timit_enroll/spk_xvector.scp',
        # './exp/xvectors_timit_eval/xvector.scp',
    ]
    
    aishell_scp_list = [
        './exp/xvectors_aishell_train/xvector.scp', 
        # './exp/xvectors_aishell_enroll/spk_xvector.scp', 
        # './exp/xvectors_aishell_eval/xvector.scp',
    ]

    with open('./trials_and_keys/domain_label.tsv', 'w') as fw:
        for scp in timit_scp_list:
            lines = open(scp, 'r').readlines()
            for line in tqdm(lines):
                uttID = line.split(' ')[0]
                fw.write(uttID +'\ttimit\n')

        for scp in aishell_scp_list:
            lines = open(scp, 'r').readlines()
            for line in tqdm(lines):
                uttID = line.split(' ')[0]
                fw.write(uttID +'\taishell\n')

def subtract_mean_and_length_normalize(xvec_file, ):
    os.system('. ./path.sh')
    os.system('')


if __name__ == '__main__':
    # # generate data for visual
    # generate_target_data(ratio=1, nTrain=300, nTest=0, nSelect=5, 
    #                      dstRoot='/mnt/d/Database/aishell_visual/wav')

    # xvecs另存为
    get_mega_xvectors(
        xvector_scp_list=['./exp/xvectors_aishell_visual/xvector.scp'], 
        dstPath='exp/mega_xvector_visual.pkl', 
    )


