import torch
import pickle
import numpy as np
import random
import datetime
import sys
import logging
import time
from torch.utils.data import DataLoader

from utils.train_utils import *

def main():

    with open('./model.info', 'r') as fr:
        modelDir = fr.readlines()[0].strip('\n')
    print(f'Model info:{modelDir}')

    from dataProcess import get_mega_xvectors

    # extract xvecs
    print('Writing xvectors to disk...')
    featDirList = [args.trainFeaDir, args.enrollFeaDir, args.evalFeaDir]
    embedDirList = [args.trainEmbedDir, args.enrollEmbedDir, args.evalEmbedDir]


    for fea, embed in zip(featDirList, embedDirList):
        args.featDir = fea
        args.embeddingDir = embed
        extract_embedding(args, modelDir)

    print('Reading xvectors from disk...')
    utt2vec_train = get_mega_xvectors([args.trainEmbedDir +'/'+ modelDir.split('/')[-1] +'/xvector.scp'])
    utt2vec_enroll = get_mega_xvectors([args.enrollEmbedDir +'/'+ modelDir.split('/')[-1] +'/xvector.scp'])
    spk2vec_enroll, spk2nutt = compute_spk_xvec(args.spk2uttEnroll, utt2vec_enroll)
    utt2vec_eval = get_mega_xvectors([args.evalEmbedDir +'/'+ modelDir.split('/')[-1] +'/xvector.scp'])

    # evaluate
    print('Scoring and evaluation ...')
    from utils.scoreGenerator import evaluate_for_all
    evaluate_for_all(utt2vec_train, spk2vec_enroll, utt2vec_eval, 
                     utt2spk_file=args.utt2spkTrain, 
                     trial_file=args.trials, 
                     evalType=args.evalType, 
                     spk2nutt=spk2nutt, 
                     apply_lda=args.ldaFlag, 
                     lda_dim=args.ldaDim, 
                     score_file='scores/scores_aishell1_{}_{}.txt'.format(args.lossType, args.evalType))


if __name__ == '__main__':

    from utils.readConf import read_tdnn_aishell_conf, getParams, prn_obj
    args = read_tdnn_aishell_conf('xi_vec.yaml')
    parser = getParams()
    args_command = parser.parse_args()
    for key, value in args_command.__dict__.items():
        if key not in args.__dict__.keys():
            print("Args from command should have defalut value in yaml")
            sys.exit(1)
        else:
            args.__dict__[key] = value
    prn_obj(args)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    main()
    