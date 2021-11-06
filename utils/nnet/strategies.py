import torch
import numpy as np
from utils.dataGenerator import get_xvec_trials_from_uttID

def NeuralPLDA_train(model, train_loader, mega_xvec_dict, num_to_id_dict, 
                     epoch, optimizer, device):

    total_loss = []
    model.train()
    for batch_idx, (data1, data2, label) in enumerate(train_loader):
        data1, data2, label = data1.to(device), data2.to(device), label.to(device)
        data1_xvec, data2_xvec = get_xvec_trials_from_uttID(
                                            mega_xvec_dict, 
                                            num_to_id_dict, 
                                            data1, data2, device
        )
        # output1, output2 = model(data1_xvec, data2_xvec)
        # criterion = ContrastiveLoss()
        # loss = criterion(output1, output2, label) 
        # total_loss.append(loss.item())
        # loss.backward()
        # optimizer.step()

        # NeuralPLDA
        output = model(data1_xvec, data2_xvec)
        loss = model.softcdet(output, label)
        total_loss.append(loss.item())
        loss.backward() # 自动生成梯度
        optimizer.step() # 更新参数

    print('Training Loss: {} after {} epochs'.format(np.mean(np.asarray(total_loss)),epoch))
    
    return np.mean(np.asarray(total_loss))

def NeuralPLDA_validate(conf, model, device, mega_xvec_dict, num_to_id_dict, 
                        data_loader, update_thresholds=False):

    model.eval() # 切换为验证模式，不学习参数
    with torch.no_grad():
        labels = torch.tensor([]).to(device)
        scores = torch.tensor([]).to(device)
        for data1, data2, label in data_loader:
            data1, data2, label = data1.to(device), data2.to(device), label.to(device) # 搬运到GPU显存
            data1_xvec, data2_xvec = load_xvec_trials_from_uttID(
                                        mega_xvec_dict, 
                                        num_to_id_dict, 
                                        data1, data2, device
            )
            labels = torch.cat((labels, label))
            scores_batch = model(data1_xvec, data2_xvec)
            scores = torch.cat((scores, scores_batch))

        cdet_mdl = model.cdet(scores, labels)
        soft_cdet_loss = model.softcdet(scores, labels)
        minc, minc_threshold = model.minc(scores, labels, update_thresholds)
    
    print('Valid set: C_det (mdl): {:.4f}'.format(cdet_mdl))
    print('Valid set: soft C_det (mdl): {:.4f}'.format(soft_cdet_loss))
    print('Valid set: C_min: {:.4f}'.format(minc))
    for beta in conf.beta:
        print('Valid set: argmin threshold [{}]: {:.4f}'.format(beta, minc_threshold[beta]))
        
    return minc, minc_threshold

def CNNModel_train():
    pass