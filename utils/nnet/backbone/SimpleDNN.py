import logging
import torch.nn as nn

class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512), 
            nn.ReLU(),
            # nn.Dropout(0.8),

            nn.Linear(512, 512), 
            nn.BatchNorm1d(512), 
            nn.ReLU(),
            # nn.Dropout(0.5), 

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # nn.Dropout(0),

            # nn.Linear(8, 2),
        )
        
        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        '''
        提取embedding
        '''
        return self.classifier(x)





