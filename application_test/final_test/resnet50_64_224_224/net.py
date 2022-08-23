import torch.nn as nn
import torch
class Net(nn.Module):
  
    def __init__(self, features):
        super(Net, self).__init__()
        '''
        self.linear_relu1 = nn.Linear(features, 128)
        self.linear_relu2 = nn.Linear(128, 256)
        self.bn1=nn.BatchNorm1d(256)
        self.linear_relu3 = nn.Linear(256, 256)
        self.bn2=nn.BatchNorm1d(256)
        self.linear_relu4 = nn.Linear(256, 256)
        self.bn3=nn.BatchNorm1d(256)
        self.linear_relu5 = nn.Linear(256, 256)
        self.bn4=nn.BatchNorm1d(256)
        self.linear_relu6 = nn.Linear(256, 256)
        self.bn5=nn.BatchNorm1d(256)
        self.linear5 = nn.Linear(256, 1)
        '''
        self.linear1=nn.Linear(features, 4*64)#进行了onehot操作
        self.linear12=nn.Linear(128, 512)
    
        self.linear21=nn.Linear(512, 1024)
        self.linear22=nn.Linear(512, 1024)
        self.linear23=nn.Linear(1024, 512)

        #兵分三路，然后contact
        self.linear3=nn.Linear(2048, 1024)
        #试一试
        self.linear4=nn.Linear(512, 64)

        self.linear5=nn.Linear(64, 1)
        
        self.maxpool= torch.nn.MaxPool1d(2, stride=2)
        #self.drop=nn.Dropout(0.1)
    def forward(self, x):
        
        #print(x[:,0].shape)[256]

        y_pred = self.linear1(x)
        y_pred = nn.functional.relu(y_pred)#变成256维度

        y_pred = self.maxpool(y_pred)#变成128维度


        y_pred = self.linear12(y_pred)
        y_pred = nn.functional.relu(y_pred)#变成512维度



        y_pred1 = self.linear21(y_pred)
        y_pred1 = nn.functional.relu(y_pred1)#变成1024维
        y_pred1 = self.maxpool(y_pred1)     #变成512维度
        y_pred1 = self.linear22(y_pred1)
        y_pred1 = nn.functional.relu(y_pred1)#变成1024维
        y_pred1 = self.linear23(y_pred1)
        y_pred1 = nn.functional.relu(y_pred1)#变成512维

        y_pred2 = self.linear21(y_pred)
        y_pred2 = nn.functional.relu(y_pred2)
        y_pred2 = self.maxpool(y_pred2)     #变成512维度
        y_pred2 = self.linear22(y_pred2)
        y_pred2 = nn.functional.relu(y_pred2)
        y_pred2 = self.linear23(y_pred2)
        y_pred2 = nn.functional.relu(y_pred2)

        y_pred3 = self.linear21(y_pred)
        y_pred3 = nn.functional.relu(y_pred3)
        y_pred3 = self.maxpool(y_pred3)     #变成512维度
        y_pred3 = self.linear22(y_pred3)
        y_pred3 = nn.functional.relu(y_pred3)
        y_pred3 = self.linear23(y_pred3)
        y_pred3 = nn.functional.relu(y_pred3)
        

        y_pred1 = torch.cat((y_pred1,y_pred2),1)
        y_pred1 = torch.cat((y_pred1,y_pred),1)
        y_pred  = torch.cat((y_pred1,y_pred3),1)#变成2048维

        y_pred = self.linear3(y_pred)
        y_pred = nn.functional.relu(y_pred)#变成1024维度


        y_pred = self.maxpool(y_pred)#变成512维度
        #考虑bridge操作吗？
        
        y_pred = self.linear4(y_pred)
        y_pred = nn.functional.relu(y_pred)#变成64维度

        y_pred = self.linear5(y_pred)
        y_pred = nn.functional.relu(y_pred)

        return y_pred


