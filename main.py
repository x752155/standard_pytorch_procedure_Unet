from data_loader import  *
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from  config import *
from loss import *
import resnet

class Model(nn.Module):
    def __init__(self,resnet):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=3,      # input height
                out_channels=8,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),      # output shape (16, 28, 28)
            nn.ReLU(),    # activation
            #nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(8,8,3,1,1)
        self.resnet =resnet



    def forward(self,x):
        layer1,layer2,layer3,layer4= self.resnet(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x

if __name__ == '__main__':
    train_data =MyDataset(txt_route+train_txt,None,is_test=False)
    test_data = MyDataset(txt_route+test_txt,None,is_test=False)
    val_data =MyDataset(txt_route+val_txt,None,is_test=False)


    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True ,num_workers=4)
    test_loader = DataLoader(dataset=test_data, batch_size=6, shuffle=False,num_workers=4)
    val_loader = DataLoader(dataset=val_data, batch_size=6, shuffle=False,num_workers=4)
    resnet_model = resnet.resnet50(pretrained=True)
    model = Model(resnet_model)
    print(model)  # net architecture
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)   # optimize all cnn parameters
    loss_module = SoftDiceLoss()



    # training and testing
    for epoch in range(config.Epoch):
        for step,(b_x, b_y) in enumerate(train_loader):   # 分配 batch data, normalize x when iterate train_loader

            b_x = Variable(b_x)
            b_y = Variable(b_y)

            output =model(b_x)               # cnn output
            loss =1-loss_module.diceCoeffv2(output, b_y,activation=None)   # cross entropy loss
            print(loss)

            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

    val_loss =0
    for step,(val_x,val_y) in enumerate(val_loader):
        val_x=  Variable(val_x)
        val_y = Variable(val_y)
        val_output = model(val_x)
        val_loss = val_loss+(1-loss_module.diceCoeffv2(val_output, val_y,activation=None))
        #这里还需要评估函数
    print(val_loss)

