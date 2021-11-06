#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from operator import pos
import numpy as np
import os
from numpy.ma.core import flatten_structured_array
import torch
from operator import itemgetter
import sys

from utils.common_utils import ivec_subtract_global_mean, ivec_normalize_length, lda_project, compute_mean, ivec_compute_lda


def evaluate_for_all(utt2vec_train, spk2vec_enroll, utt2vec_eval, 
                     utt2spk_file, trial_file, evalType='plda', 
                     spk2nutt={}, apply_lda=False, lda_dim=200, 
                     score_file='exp/test/scores'):
    '''
    直接输入特征向量，得到最终评价结果
    utt2vec_train, spk2vec_enroll, utt2vec_eval: dict
    utt2spk_file属于train set；spk2nutt_file属于enroll set；
    '''

    # convert utt2spk & spk2nutt
    utt2spk_dict = {}
    with open(utt2spk_file, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            utt = line.strip().split(' ')[0]
            spk = line.strip().split(' ')[1]
            utt2spk_dict[utt] = spk

    if spk2nutt == {}:
        print('Noting: each piece of enrollmant has only one utt !!!')
        for key, _ in spk2vec_enroll.items():
            spk2nutt[key] = 1
        

    # centralization
    mean = compute_mean(utt2vec_train)
    utt2vec_centered = ivec_subtract_global_mean(utt2vec_train, mean)

    if apply_lda:
        # LDA降维
        LDA = ivec_compute_lda(utt2spk_file, utt2vec_centered, lda_dim)
        utt2vec_lda = lda_project(utt2vec_centered, LDA)
        # length normalization
        utt2vec_norm = ivec_normalize_length(utt2vec_lda)
    else:
        LDA = []
        utt2vec_norm = ivec_normalize_length(utt2vec_centered)
    
    # compute scores
    if evalType == 'plda':
        from utils.plda_lda import compute_plda
        PLDA = compute_plda(utt2spk_dict, utt2vec_norm, 'exp/test/plda.python')
        plda_scoring(spk2vec_enroll, spk2nutt, utt2vec_eval, trial_file, 
                    score_file, mean, PLDA, apply_lda=apply_lda, LDA=LDA) 
    elif evalType == 'cosine':
        cosine_scoring(spk2vec_enroll, utt2vec_eval, trial_file, score_file, 
                       mean, apply_lda=apply_lda, LDA=LDA)
    
    # evaluation
    eer, _ = compute_eer(trial_file, score_file)
    min_dcf, _ = compute_min_dcf(trial_file, score_file, p_target=0.01)
    print("\t\t\t\t\tEER={}%, minDCF={}".format(round(eer, 4), round(min_dcf, 4))
    )


def plda_scoring(spk2vec_enroll, spk2nutt, utt2vec_eval, trials_file, 
                 scores_file, mean, PLDA, apply_lda=False, LDA=[]):

    if apply_lda:
        spk2vec_enroll_pre = ivec_normalize_length(lda_project(ivec_subtract_global_mean(spk2vec_enroll, mean), LDA))
        utt2vec_eval_pre = ivec_normalize_length(lda_project(ivec_subtract_global_mean(utt2vec_eval, mean), LDA))
    else:
        spk2vec_enroll_pre = ivec_normalize_length(ivec_subtract_global_mean(spk2vec_enroll, mean))
        utt2vec_eval_pre = ivec_normalize_length(ivec_subtract_global_mean(utt2vec_eval, mean))
    
    fw = open(scores_file, 'w')
    with open(trials_file, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            enroll_ID = line.strip().split(' ')[0]
            eval_ID = line.strip().split(' ')[1]
            transform_enroll = PLDA.transform_ivector(
                                        spk2vec_enroll_pre[enroll_ID], 
                                        int(spk2nutt[enroll_ID]))
            transform_eval = PLDA.transform_ivector(utt2vec_eval_pre[eval_ID], 1)
            score = PLDA.log_likelihood_ratio(transform_enroll, 
                                              int(spk2nutt[enroll_ID]), 
                                              transform_eval)
            fw.write(enroll_ID +' '+ eval_ID +' '+ str(score[0][0]) +'\n')
    fw.close()


def cosine_scoring(spk2vec_enroll, utt2vec_eval, trials_file, scores_file, 
                   mean, apply_lda=False, LDA=[]):
    
    if apply_lda:
        spk2vec_enroll_pre = ivec_normalize_length(lda_project(ivec_subtract_global_mean(spk2vec_enroll, mean), LDA))
        utt2vec_eval_pre = ivec_normalize_length(lda_project(ivec_subtract_global_mean(utt2vec_eval, mean), LDA))
    else:
        spk2vec_enroll_pre = ivec_normalize_length(ivec_subtract_global_mean(spk2vec_enroll, mean))
        utt2vec_eval_pre = ivec_normalize_length(ivec_subtract_global_mean(utt2vec_eval, mean))

    fw = open(scores_file, 'w')

    with open(trials_file, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            enroll_ID = line.strip().split(' ')[0]
            eval_ID = line.strip().split(' ')[1]
            xvec1 = spk2vec_enroll_pre[enroll_ID]
            xvec2 = utt2vec_eval_pre[eval_ID]
            score = np.dot(xvec1, xvec2) / (np.linalg.norm(xvec1) * np.linalg.norm(xvec2))
            fw.write(enroll_ID +' '+ eval_ID +' '+ str(score) +'\n')

    fw.close()



def prepare_for_eer(trials_file, scores_file):
    '''
    Copied from egs/sre10/v1/local/prepare_for_eer.py (commit 9cb4c4c2fb0223ee90c38d98af11305074eb7ef8)
    
    Given a trials and scores file, this script
    prepares input for the binary compute-eer.
    '''
    trials = open(trials_file, 'r').readlines()
    scores = open(scores_file, 'r').readlines()
    spkrutt2target = {} # label字典
    for line in trials:
        spkr, utt, target = line.strip().split()
        spkrutt2target[spkr+utt] = target
    score_list = []
    label_list = []
    for line in scores:
        spkr, utt, score = line.strip().split()
        score_list.append(float(score))
        label_list.append(spkrutt2target[spkr+utt])

    return label_list, score_list


def compute_eer(trials_file, scores_file, positive_label='target'):
    """Compute eer
    labels: ground-truth label, should be a 1-d list or np.array, each element represents the ground-truth label of one sample
    scores: model prediction, should be a 1-d list or np.array, each element represents the model prediction of one sample
    positive_label: the class that is viewed as positive class when computing EER
    return: equal error rate (EER)
    Based on: https://blog.csdn.net/zjm750617105/article/details/52558779
    """
    labels, scores = prepare_for_eer(trials_file, scores_file)
    
    target_scores = []
    nontarget_scores = []
    
    for (label, score) in zip(labels, scores):
        if label == positive_label:
            target_scores.append(score)
        else:
            nontarget_scores.append(score)

    target_scores = sorted(target_scores)
    nontarget_scores = sorted(nontarget_scores)

    target_size = len(target_scores)
    target_position = 0
    for target_position in range(target_size):
        nontarget_size = len(nontarget_scores)
        nontarget_n = nontarget_size * target_position * 1.0 / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)
        if nontarget_position < 0:
            nontarget_position = 0
        if nontarget_scores[nontarget_position] < target_scores[target_position]:
            # print("nontarget_scores[nontarget_position] is {} {}".format(nontarget_position, nontarget_scores[nontarget_position]))
            # print("target_scores[target_position] is {} {}".format(target_position, target_scores[target_position]))
            break

    threshold = target_scores[target_position]
    # print("threshold is --> {}".format(threshold))
    eer = target_position * 1.0 / target_size * 100
    # print("eer is --> {}% at threshold {}".format(round(eer, 5), threshold))
    return eer, threshold


def compute_min_dcf(trials_file, scores_file, p_target, c_miss=1, c_fa=1):
    """
    trials_file: Input trials file, with columns of the form <utt1> <utt2> <target/nontarget>
    scores_file: Input scores file, with columns of the form <utt1> <utt2> <score>
    p_target: The prior probability of the target speaker in a trial.
    c-miss: Cost of a missed detection.  This is usually not changed.
    c-fa: Cost of a spurious detection.  This is usually not changed.
    Based on: https://github.com/kaldi-asr/kaldi/blob/cbed4ff688a172a7f765493d24771c1bd57dcd20/egs/sre08/v1/sid/compute_min_dcf.py
    """
    # 创建3个列表，一个是错误拒绝率，一个是错误接受率，一个是得到上述错误率对应的决策阈值
    def ComputeErrorRates(scores, labels):
        # 将得分从小到大进行排序，并得到对应的索引 We will treat the sorted scores as the
        # thresholds at which the the error-rates are evaluated.
        sorted_indexes, thresholds = zip(*sorted(
            [(index, threshold) for index, threshold in enumerate(scores)],
            key=itemgetter(1))
            )
        sorted_labels = []
        labels = [labels[i] for i in sorted_indexes]
        fnrs = [] # false negative rates
        fprs = [] # false positive rates

        # fnrs[i] is the number of errors made by incorrectly 
        # rejecting scores less than thresholds[i]. 
        # fprs[i] is the total number of times that we have 
        # correctly accepted scores greater than thresholds[i].
        for i in range(0, len(labels)):
            if i == 0:
                fnrs.append(labels[i])
                fprs.append(1 - labels[i])
            else:
                fnrs.append(fnrs[i-1] + labels[i])
                fprs.append(fprs[i-1] + 1 - labels[i])
        fnrs_norm = sum(labels) # 正样本总数
        fprs_norm = len(labels) - fnrs_norm

        # Now divide by the total number of false negative errors to
        # obtain the false positive rates across all thresholds
        fnrs = [x / float(fnrs_norm) for x in fnrs]

        # Divide by the total number of corret positives to get the
        # true positive rate.  Subtract these quantities from 1 to
        # get the false positive rates.
        fprs = [1 - x / float(fprs_norm) for x in fprs]
        return fnrs, fprs, thresholds
    
    # 计算检测代价函数的最小值，参考NIST 2016评价指标
    def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
        min_c_det = float("inf")
        min_c_det_threshold = thresholds[0]
        for i in range(0, len(fnrs)):
            # See Equation (2).  it is a weighted sum of false negative
            # and false positive errors.
            c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
            if c_det < min_c_det:
                min_c_det = c_det
                min_c_det_threshold = thresholds[i]
        # See Equations (3) and (4). 
        c_def = min(c_miss * p_target, c_fa * (1 - p_target))
        min_dcf = min_c_det / c_def # 归一化cost
        
        return min_dcf, min_c_det_threshold

    
    scores_file = open(scores_file, 'r').readlines()
    trials_file = open(trials_file, 'r').readlines()

    trials = {}
    for line in trials_file:
        utt1, utt2, target = line.rstrip().split()
        trial = utt1 + " " + utt2
        trials[trial] = target

    scores = []
    labels = []
    for line in scores_file:
        utt1, utt2, score = line.rstrip().split()
        trial = utt1 + " " + utt2
        if trial in trials:
            scores.append(float(score))
            if trials[trial] == "target":
                labels.append(1)
            else:
                labels.append(0)
        else:
            raise Exception("Missing entry for " + utt1 + " and " + utt2
                + " " + scores_file)

    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
    mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, 
                                      p_target, c_miss, c_fa
    )
    # sys.stdout.write("{0:.4f}\n".format(mindcf))
    # txt = "minDCF(p-target={2}, c-miss={3}, c-fa={4}): {0:.4f} at threshold {1:.4f} \n"
    # txt = "minDCF(p-target={2}): {0:.4f} at threshold {1:.4f} \n"
    # sys.stderr.write(txt.format(mindcf, threshold, p_target, c_miss, c_fa))
    return mindcf, threshold


def generate_scores_for_Nplda(score_file, trials_file, mega_dict, 
                              model, device, batch_size=102400):
    
    model = model.to(torch.device('cpu'))
    trials = np.genfromtxt(trials_file, dtype='str')[:,:2]
    iters = len(trials) // batch_size 
    S = torch.tensor([])
    model = model.eval()
    from utils.dataGenerator import get_xvec_trials_from_list
    with torch.no_grad():
        for i in range(iters+1):
            x1_b, x2_b = get_xvec_trials_from_list(
                        mega_dict, 
                        trials[i*batch_size : (i+1)*batch_size], 
                        device=torch.device('cpu')
            )
            # NPLDA
            # S_b, _ = model.forward(x1_b, x2_b)
            # S_b, _ = model.test(x1_b, x2_b) # selected

            # S_b = model.domain_test(x1_b, x2_b) # 异域：替换全局均值
            S_b = model.forward(x1_b, x2_b) # 同域
            S = torch.cat((S, S_b))
        scores = np.asarray(S.detach()).astype(str)
    
    if not os.path.exists(os.path.dirname(score_file)):
        os.makedirs(os.path.dirname(score_file))
    np.savetxt(score_file, np.c_[trials, scores], 
               fmt='%s', delimiter='\t', comments='')
    model = model.to(device)


if __name__ == '__main__':
    trials_file = './trials_timit'
    scores_file = './scores/06250926/epoch50.txt'
    eer = compute_eer(trials_file, scores_file, positive_label='target')
    print(eer)



