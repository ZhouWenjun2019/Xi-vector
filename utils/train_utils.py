from time import sleep
import torch
import argparse
from datetime import datetime
import os
import kaldi_python_io
from kaldiio import ReadHelper
from torch.utils.data import IterableDataset
import glob
import sys
from collections import OrderedDict
from torch.multiprocessing import Pool
import numpy as np

from utils.nnet.models import *


class nnet3EgsDL(IterableDataset):
    """ 
    Data loader class to read directly from egs files, no HDF5
    """

    def __init__(self, arkFile):
        self.fid = kaldi_python_io.Nnet3EgsReader(arkFile)

    def __iter__(self):
        return iter(self.fid)


def prepareModel(args):
    '''
    参考pytorch_xvectors
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print('Initializing Model..')
    net = eval(f'{args.modelType}({args.numSpkrs}, {args.featDim}, p_dropout=0)')
    optimizer = torch.optim.Adam(net.parameters(), lr=args.baseLR)

    net.to(device)

    eventID = datetime.now().strftime('%Y%m%d%H%M')
    saveDir = './save_models/{}_{}_{}'.format(args.modelType, args.lossType, eventID)
    os.makedirs(saveDir)
    
    return net, optimizer, saveDir


def computeValidAccuracy(args, modelDir):
    """
    Computes frame-level validation accruacy
    """
    import glob
    modelFile = max(glob.glob(modelDir+'/*'), key=os.path.getctime)
    # Load the model
    net = eval(f'{args.modelType}({args.numSpkrs}, {args.featDim}, p_dropout=0)')
    if args.lossType == 'softmax':
        from utils.nnet.margins import InnerProduct
        margin = InnerProduct(512, args.numSpkrs)
    if args.lossType == 'arcface':
        from utils.nnet.margins import ArcMarginProduct
        margin = ArcMarginProduct(512, args.numSpkrs, s=30, m=0.5)
    if args.lossType == 'cosface':
        from utils.nnet.margins import AddMarginProduct
        margin = AddMarginProduct(512, args.numSpkrs, s=30, m=0.5)

    checkpoint = torch.load(modelFile, map_location=torch.device('cuda'))
    from collections import OrderedDict
    net_state_dict, margin_state_dict = OrderedDict(), OrderedDict()
    for k, v in checkpoint['net_state_dict'].items():
        if k.startswith('module.'):
            net_state_dict[k[7:]] = v  # ugly fix to remove 'module' from key
        else:
            net_state_dict[k] = v
    for k, v in checkpoint['margin_state_dict'].items():
        if k.startswith('module.'):
            margin_state_dict[k[7:]] = v  # ugly fix to remove 'module' from key
        else:
            margin_state_dict[k] = v
    
    # load params
    net.load_state_dict(net_state_dict)
    margin.load_state_dict(margin_state_dict)
    net = net.cuda(); margin = margin.cuda()
    net.eval(); margin.eval()

    correct, incorrect = 0, 0
    for validArk in glob.glob(args.egDir +'/valid_egs.*.ark'):
        x = kaldi_python_io.Nnet3EgsReader(validArk)
        for key, mat in x:
            logit = net(x=torch.Tensor(mat[0]['matrix']).permute(1,0).unsqueeze(0).cuda(), eps=0)
            out = margin(logit, torch.Tensor([mat[1]['matrix'][0][0][0]]).cuda())
            if mat[1]['matrix'][0][0][0]+1 == torch.argmax(out)+1:
                correct += 1
            else:
                incorrect += 1

    print(f'Number of valid samples: {correct+incorrect}')
    return 100.0*correct/(correct+incorrect)

def par_core_extractXvectors(inFeatsScp, outXvecArk, outXvecScp, net, layerName):
    """ 
    To be called using pytorch multiprocessing
    Note: This function reads all the data from feats.scp into memory before 
    inference. Hence, make sure the file is not too big (Hint: use
    split_data_dir.sh)
    """

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    eval('net.%s.register_forward_hook(get_activation(layerName))' %layerName)

    with kaldi_python_io.ArchiveWriter(outXvecArk, outXvecScp) as writer:
        with ReadHelper('scp:%s'%inFeatsScp) as reader:
            for key, mat in reader:
                # out = net(x=torch.Tensor(mat).permute(1,0).unsqueeze(0).cuda(),
                #           eps=0)
                # 必须先跑完整个网络，才能有fc1处的结果
                out = net(x=torch.Tensor(mat).permute(1,0).unsqueeze(0), eps=0)
                writer.write(key, np.squeeze(activation[layerName].cpu().numpy()))

def extract_embedding(args, modelDir):
    def getSplitNum(text):
        return int(text.split('/')[-1].lstrip('split'))

    # Checking for input features and splitN directories
    try:
        nSplits = int(sorted(glob.glob(args.featDir+'/split*'),
                  key=getSplitNum)[-1].split('/')[-1].lstrip('split'))
    except ValueError:
        print('[ERROR] Cannot find %s/splitN directory' %args.featDir)
        print('Use utils/split_data.sh to create this directory')
        sys.exit(1)

    if not os.path.isfile('%s/split%d/1/feats.scp' %(args.featDir, nSplits)):
        print('Cannot find input features')
        sys.exit(1)

    # Check for trained model
    try:
        modelFile = max(glob.glob(modelDir+'/*.tar'), key=os.path.getctime)
    except ValueError:
        print("[ERROR] No trained model has been found in {}.".format(modelDir))
        sys.exit(1)

    # Load model definition
    net = eval(f'{args.modelType}({args.numSpkrs}, {args.featDim}, p_dropout=0)')
    # model = eval(f'TDNN_FC(\'{args.modelType}\', {args.numSpkrs}, \'{args.lossType}\')')

    checkpoint = torch.load(modelFile, map_location=torch.device('cuda'))
    net_state_dict = OrderedDict()
    if 'relation' in args.modelType:
        checkpoint_dict = checkpoint['encoder_state_dict']
    else:
        checkpoint_dict = checkpoint['net_state_dict']
    for k, v in checkpoint_dict.items():
        if k.startswith('module.'):
            net_state_dict[k[7:]] = v  # ugly fix to remove 'module' from key
        else:
            net_state_dict[k] = v

    # load trained weights
    net.load_state_dict(net_state_dict)
    net.eval()

    embeddingDir = args.embeddingDir +'/'+ modelDir.split('/')[-1]
    print('Save xvectors to {}'.format(embeddingDir))
    if not os.path.isdir(embeddingDir):
        os.makedirs(embeddingDir)

    print('Extracting xvectors by distributing jobs to pool workers... ')
    if not args.nProcs:
        args.nProcs = nSplits

    L = [('%s/split%d/%d/feats.scp' %(args.featDir, nSplits, i),
          '%s/xvector.%d.ark' %(embeddingDir, i),
          '%s/xvector.%d.scp' %(embeddingDir, i), net, args.layerName) for i in range(1, nSplits+1)]
    pool2 = Pool(processes=args.nProcs)
    result = pool2.starmap(par_core_extractXvectors, L)
    pool2.terminate() # 直接关闭
    print('Multithread job has been finished.')

    print(f'Writing xvectors to {embeddingDir}')
    os.system(f'cat {embeddingDir}/xvector.*.scp > {embeddingDir}/xvector.scp')

    return embeddingDir

def compute_spk_xvec(spk2utt, utt2vec):

    spk_xvec, num_utts = {}, {}

    with open(spk2utt, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            spk = line.strip('\n').split(' ')[0]
            uttList = line.strip('\n').split(' ')[1:]
            num_utts[spk] = len(uttList)

            xvecList = []
            for utt in uttList:
                xvecList.append(utt2vec[utt])
            
            spk_xvec[spk] = np.mean(xvecList, axis=0)

    return spk_xvec, num_utts




if __name__ == '__main__':
    # parser = getParams()
    pass
