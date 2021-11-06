#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import argparse

class read_tdnn_aishell_conf:
    '''
    配置文件参数读取
    '''
    def __init__(self, configfile):
        conf = yaml.safe_load(open(configfile, 'r'))

        # General Parameters
        self.modelType = conf['modelType']
        self.egDir = conf['egDir']

        # Training Parameters
        self.numSpkrs = int(conf['numSpkrs'])
        self.numArchives = int(conf['numArchives'])
        self.numEgsPerArk = int(conf['numEgsPerArk'])
        self.featDim = int(conf['featDim'])
        self.batchSize = int(conf['batchSize'])
        self.logStepSize = int(conf['logStepSize'])
        self.lossType = conf['lossType']

        # Optimization Params
        self.baseLR = float(conf['baseLR'])
        self.maxLR = float(conf['maxLR'])
        self.numEpochs = int(conf['numEpochs'])
        self.noiseEps = float(conf['noiseEps'])
        self.preFetchRatio = int(conf['preFetchRatio'])
        self.stepFrac = float(conf['stepFrac'])
        self.pDropMax = float(conf['pDropMax'])

        # Extraction Params
        self.layerName = conf['layerName']
        self.nProcs = int(conf['nProcs'])
        self.featDir = ''
        self.embeddingDir = ''
        self.trainFeaDir = conf['trainFeaDir']
        self.trainEmbedDir = conf['trainEmbedDir']
        self.enrollFeaDir = conf['enrollFeaDir']
        self.enrollEmbedDir = conf['enrollEmbedDir']
        self.evalFeaDir = conf['evalFeaDir']
        self.evalEmbedDir = conf['evalEmbedDir']

        # Scoring Params
        self.utt2spkTrain = conf['utt2spkTrain']
        self.spk2uttEnroll = conf['spk2uttEnroll']
        self.spk2nuttEnroll = conf['spk2nuttEnroll']
        self.trials = conf['trials']
        self.ldaFlag = bool(conf['ldaFlag'])
        self.ldaDim = int(conf['ldaDim'])
        self.evalType = conf['evalType']

  
def getParams():
    '''
    命令行参数读取
    '''
    parser = argparse.ArgumentParser()

    # Training Parameters
    trainingArgs = parser.add_argument_group('General Training Parameters')
    trainingArgs.add_argument('--lossType', default='softmax',  
                              help='Loss function of classification')

    # Scoring Params
    scoringArgs = parser.add_argument_group('General scoring Parameters')
    scoringArgs.add_argument('--ldaFlag', action='store_true', 
                             help='Reduce dimension of xvecs')
    scoringArgs.add_argument('--ldaDim', default=200, type=int, 
                             help='Reduce dimension of xvecs')
    scoringArgs.add_argument('--evalType', default='plda', 
                             help='Eval type for speaker verification')
    

    return parser


def prn_obj(obj):
    '''
    打印参数
    '''
    print('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))

