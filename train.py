import argparse
import torch
from torch import optim
import pickle
import numpy as np
import random
import datetime
import sys
import logging
import time
from torch.utils.data import DataLoader

from utils.train_utils import *

def train():
    
    net, optimizer, saveDir = prepareModel(args)
    net.train()

    if args.lossType == 'softmax':
        from utils.nnet.margins import InnerProduct
        margin = InnerProduct(512, args.numSpkrs).to(device)
    elif args.lossType == 'arcface':
        from utils.nnet.margins import ArcMarginProduct
        margin = ArcMarginProduct(512, args.numSpkrs, s=30, m=0.5).to(device) 
    elif args.lossType == 'cosface':
        from utils.nnet.margins import AddMarginProduct
        margin = AddMarginProduct(512, args.numSpkrs, s=30, m=0.5).to(device)
    
    optimizer.add_param_group({'params': margin.parameters(), 'lr': args.baseLR})


    totalSteps = args.numEpochs * args.numArchives
    numBatchesPerArk = int(args.numEgsPerArk/args.batchSize)

    from torch.optim.lr_scheduler import OneCycleLR
    circle_lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=args.maxLR,
            cycle_momentum=False,
            div_factor=5,
            final_div_factor=1e+3,
            pct_start=0.15, 
            total_steps=totalSteps*numBatchesPerArk, # 总批数
    )

    criterion = torch.nn.CrossEntropyLoss().to(device)

    eps = args.noiseEps

    step = 0
    while step < totalSteps: # 以ark为单位
        # load data
        archiveI = step%args.numArchives + 1 # 依次训练
        archive_start_time = time.time()
        ark_file = f'{args.egDir}/egs.{archiveI}.ark'
        print('Reading from archive %d' %archiveI)

        preFetchRatio = args.preFetchRatio

        egs = nnet3EgsDL(ark_file)
        data_loader = DataLoader(
            egs, 
            batch_size=preFetchRatio*args.batchSize, # 30*32
            shuffle=False, 
            num_workers=0, 
            drop_last=False, 
            pin_memory=True
        )

        batchI, loggedBatch = 0, 0
        loggingLoss = 0.0 # 做记录使用的损失值
        start_time = time.time()
        for _, (X, Y) in data_loader:
            Y = Y['matrix'][0][0][0].to(device) # 960=30*32
            X = X['matrix'].to(device) # 960*213*30
            try:
                assert max(Y) < args.numSpkrs and min(Y) >= 0
            except:
                print('Read an out of range value at iter %d' %iter)
                continue
            if torch.isnan(X).any():
                print('Read a nan value at iter %d' %iter)
                continue

            accumulateStepSize = 4 
            preFetchBatchI = 0
            while preFetchBatchI <= int(len(Y)/args.batchSize) - accumulateStepSize:
                # Accumulated gradients used
                optimizer.zero_grad()
                for _ in range(accumulateStepSize): # 累积多批梯度再更新
                    # fwd + bckwd + optim
                    X_sub = X[preFetchBatchI*args.batchSize : (preFetchBatchI+1)*args.batchSize,:,:].permute(0,2,1) # 32*30*213
                    Y_sub = Y[preFetchBatchI*args.batchSize : (preFetchBatchI+1)*args.batchSize].squeeze()
                    raw_logits = net(X_sub, eps) # 出问题
                    output = margin(raw_logits, Y_sub)
                    loss = criterion(output, Y_sub)

                    if np.isnan(loss.item()):
                        print('Nan encountered at iter %d. Exiting...' %iter)
                        sys.exit(1)
                    
                    loss.backward()
                    loggingLoss += loss.item()
                    batchI += 1
                    preFetchBatchI += 1

                # Does the update
                optimizer.step() 
                circle_lr_scheduler.step()

                # Log
                if batchI-loggedBatch >= args.logStepSize:
                    logStepTime = time.time() - start_time
                    print('Batch: (%d/%d) Avg Time/batch: %1.3f Avg Loss/batch: %1.3f' %(
                        batchI,
                        numBatchesPerArk,
                        logStepTime/(batchI-loggedBatch),
                        loggingLoss/(batchI-loggedBatch))
                    )
                    loggingLoss = 0.0
                    start_time = time.time()
                    loggedBatch = batchI

        print('Archive processing time: %1.3f' %(time.time()-archive_start_time))
        
        # Update dropout
        if 1.0*step < args.stepFrac*totalSteps:
            p_drop = args.pDropMax*step/(args.stepFrac*totalSteps) # 线性递增
        else:
            p_drop = max(0, args.pDropMax*(2*step - totalSteps*(args.stepFrac+1))/(totalSteps*(args.stepFrac-1))) # fast decay
        for x in net.modules():
            if isinstance(x, torch.nn.Dropout):
                x.p = p_drop
        print('Dropout updated to %f' %p_drop)

        # Save checkpoint
        torch.save({
            'step': step,
            'archiveI':archiveI,
            'net_state_dict': net.state_dict(),
            'margin_state_dict': margin.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'args': args,
            }, '{}/checkpoint_step{}.tar'.format(saveDir, step))

        # Compute validation loss, update LR if using plateau rule
        valAcc = computeValidAccuracy(args, saveDir)
        print('Validation accuracy is %1.2f%%' %(valAcc))

        # Cleanup. We always retain the last 10 models
        if step > 10:
            if os.path.exists('%s/checkpoint_step%d.tar' %(saveDir, step-10)):
                os.remove('%s/checkpoint_step%d.tar' %(saveDir, step-10))
        step += 1

    with open('./model.info', 'w') as fw:
        fw.write(saveDir +'\n')
    print(f'Saved model to {saveDir}')


def temp():
    fr = open('/home/zhouwj/kaldi/egs/voxceleb/v2/exp/xvector_nnet_1a/xvectors_voxceleb1_test/xvector.scp', 'r').readlines()
    fw = open('/home/zhouwj/kaldi/egs/voxceleb/v2/exp/xvector_nnet_1a/xvectors_voxceleb1_test/num_id.ark', 'w')
    for line in fr:
        id = line.split(' ')[0]
        num = '1'
        fw.write(id +' '+ num +'\n')
    fw.close()


if __name__ == '__main__':

    from utils.readConf import read_tdnn_aishell_conf, getParams, prn_obj
    args = read_tdnn_aishell_conf('xi_vec.yaml')
    parser = getParams() # 命令行覆盖
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

    train()




