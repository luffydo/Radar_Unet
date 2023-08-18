import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from rodnet.models.modules.cbam import CBAM
from rodnet.models.DANet import DAHead_Position,DAHead_Channel,DAHead_backbone64,DAHead_backbone128,DAHead_backbone192,DAHead_backbone384,DAHead_backbone256
class MNet(nn.Module):
    def __init__(self):
        super(MNet, self).__init__()
      
        

    def forward(self, x):
        x = x
        return x,x,x,x,x
class MNet1(nn.Module):
    def __init__(self):
        super(MNet1, self).__init__()
      
        

    def forward(self, x, out1 ,out2):
        x = x
        out1 = out1
        out2 = out2
        return x,out1,out2,x,x


def unsample1(x):
    return F.interpolate(x,scale_factor=2, mode='bilinear')
def unsample(x):
    return F.interpolate(x,scale_factor=2, mode='trilinear')
def unsample_out1(x):
    return F.interpolate(x,scale_factor=4, mode='bilinear')
def unsample_out2(x):
    return F.interpolate(x,scale_factor=2, mode='bilinear')   
class BackboneNet_MAX_V3_1(nn.Module):
    def __init__(self):
        super(BackboneNet_MAX_V3_1, self).__init__()
        self.conv0 = nn.Conv3d(2, 16, (1, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv1 = nn.Conv3d(16, 32, (4, 4, 4), stride=2, padding=(1, 1, 1))

        self.conv_a = nn.Conv3d(16, 16, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_a1 = nn.Conv3d(16, 16, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_b = nn.Conv3d(32, 32, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_b1 = nn.Conv3d(32, 32, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_c = nn.Conv3d(64, 64, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_c1 = nn.Conv3d(64, 64, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_d = nn.Conv2d(64, 64, ( 3, 3), stride=1, padding=( 1, 1))
        self.conv_d1 = nn.Conv2d(64, 32, ( 3, 3), stride=1, padding=( 1, 1))
        self.conv_e = nn.Conv2d(64, 32, (3, 3), stride=1, padding=( 1, 1))
        self.conv_e1 = nn.Conv2d(32, 16, ( 3, 3), stride=1, padding=( 1, 1))
               
        self.conv2 = nn.Conv3d(32, 64, (6, 4, 4), stride=2, padding=(2, 1, 1))

        self.relu = nn.PReLU()


        # self.deconv2 = F.interpolate(size=(2,64,64), scale_factor=2, mode='bilinear')
        # self.deconv3 = F.interpolate(input,size=(4,128,128), scale_factor=2, mode='bilinear')


        # self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1))
        self.maxpoolc = nn.MaxPool3d(kernel_size=(2, 1, 1))

        self.head = DAHead_backbone64(in_channels=64,nclass=64)
        self.conv6 = nn.Conv2d(32,32,( 3, 3), stride=1, padding=( 1, 1))
        self.conv7 = nn.Conv2d(32,32,( 3, 3), stride=1, padding=( 1, 1))
        self.conv8 = nn.Conv2d(64,16,( 3, 3), stride=1, padding=( 1, 1))
        self.conv9 = nn.Conv2d(16,3,( 1, 1), stride=1, padding=( 0, 0))
        # self.convk = nn.Conv2d(16,32,( 3, 3), stride=1, padding=( 1, 1))
        self.sigm = nn.Sigmoid()
    def forward(self, x):
        # batch_size, in_chirps, n_channels, w, h = x.shape
        
        x = self.conv0(x)
        b = x
        x = self.relu(x) 
        
        x = self.conv_a(x)         #    (N, 16, 4, 128, 128)
        x = self.relu(x)
        x = self.conv_a1(x)
        x = x + b                    #res 
        x = self.relu(x)            #(N, 16, 4, 128, 128)
        
        x = self.conv1(x)           # (N, 32, 2, 64, 64)
        c = x
        
        x = self.conv_b(x)
        x = self.relu(x)
        x = self.conv_b1(x)
        x = x + c
        x = self.relu(x)        # (N, 32, 2, 64, 64)
        
        x = self.conv2(x)           # (N, 64, 1, 32, 32)
        d = x
        
        x = self.conv_c(x)
        x = self.relu(x)
        x = self.conv_c1(x)
        x = x + d
        x = self.relu(x)        # (N, 64, 1, 32, 32)
  
        x = x.view(-1, 64, 32, 32)
        x = self.head(x)        # (N, 64, 32, 32)


        x = unsample1(x)                 # (N, 64, 64, 64)

        
        x = self.conv_d(x)
        x = self.relu(x)
        x = self.conv_d1(x)
        c = (self.maxpoolc(c)).view(-1, 32, 64, 64)
        x = torch.cat((x,c),dim=1)
        x = self.relu(x)          #(N, 64,  64, 64)  
        
        x = unsample1(x)         #(N, 64,  128, 128) 
        
        x = self.conv_e(x)
        x = self.relu(x)
        x = self.conv_e1(x)
        b = (self.maxpool(b)).view(-1, 16, 128, 128)
        x = torch.cat((x,b),dim=1)
        x = self.relu(x)          #(N, 32, 128, 128) 
        
        # x = self.convk(x)
        # x = self.relu(x)        #(N, 32, 128, 128)
        
        b = x
        x = self.conv6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = torch.cat((x,b),dim=1)
        x = self.relu(x)          #(N, 64, 128, 128) 
        
        x = self.conv8(x)       #(N, 32, 128, 128) 
        x = self.conv9(x)       #(N, 16, 128, 128)
        x = self.sigm(x)        #(N, 3, 128, 128)
        return x   

class BackboneNet_MAX_V5_1(nn.Module):
    def __init__(self):
        super(BackboneNet_MAX_V5_1, self).__init__()
        self.conv0 = nn.Conv3d(2, 16, (1, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv1 = nn.Conv3d(16, 32, (4, 4, 4), stride=2, padding=(1, 1, 1))

        self.conv_a = nn.Conv3d(16, 16, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_a1 = nn.Conv3d(16, 16, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_b = nn.Conv3d(32, 32, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_b1 = nn.Conv3d(32, 32, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_c = nn.Conv3d(64, 64, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_c1 = nn.Conv3d(64, 64, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_c2 = nn.Conv3d(128, 128, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_c3 = nn.Conv3d(128, 128, (1, 3, 3), stride=1, padding=(0, 1, 1))        
        self.conv_c4 = nn.Conv2d(128, 128, ( 3, 3), stride=1, padding=( 1, 1))
        self.conv_c5 = nn.Conv2d(128, 64, ( 3, 3), stride=1, padding=( 1, 1))         
        
        self.conv_d = nn.Conv2d(128, 64, ( 3, 3), stride=1, padding=( 1, 1))
        self.conv_d1 = nn.Conv2d(64, 32, ( 3, 3), stride=1, padding=( 1, 1))
        self.conv_e = nn.Conv2d(64, 32, (3, 3), stride=1, padding=( 1, 1))
        self.conv_e1 = nn.Conv2d(32, 16, ( 3, 3), stride=1, padding=( 1, 1))
               
        self.conv2 = nn.Conv3d(32, 64, (2, 4, 4), stride=2, padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(64, 128, (2, 4, 4), stride=2, padding=(0, 1, 1))
        
        self.relu = nn.PReLU()


        # self.deconv2 = F.interpolate(size=(2,64,64), scale_factor=2, mode='bilinear')
        # self.deconv3 = F.interpolate(input,size=(4,128,128), scale_factor=2, mode='bilinear')


        # self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1))
        self.maxpoolc = nn.MaxPool3d(kernel_size=(2, 1, 1))

        self.head = DAHead_backbone64(in_channels=64,nclass=64)
        self.head1 = DAHead_backbone128(in_channels=128,nclass=128)
        self.conv6 = nn.Conv2d(32,32,( 3, 3), stride=1, padding=( 1, 1))
        self.conv7 = nn.Conv2d(32,32,( 3, 3), stride=1, padding=( 1, 1))
        self.conv8 = nn.Conv2d(64,16,( 3, 3), stride=1, padding=( 1, 1))
        self.conv9 = nn.Conv2d(16,3,( 1, 1), stride=1, padding=( 0, 0))
        # self.convk = nn.Conv2d(16,32,( 3, 3), stride=1, padding=( 1, 1))
        self.sigm = nn.Sigmoid()
    def forward(self, x):
        # batch_size, in_chirps, n_channels, w, h = x.shape
        
        x = self.conv0(x)
        b = x
        x = self.relu(x) 
        
        x = self.conv_a(x)         #    (N, 16, 4, 128, 128)
        x = self.relu(x)
        x = self.conv_a1(x)
        x = x + b                    #res 
        x = self.relu(x)            #(N, 16, 4, 128, 128)
        
        x = self.conv1(x)           # (N, 32, 2, 64, 64)
        c = x
        
        x = self.conv_b(x)
        x = self.relu(x)
        x = self.conv_b1(x)
        x = x + c
        x = self.relu(x)        # (N, 32, 2, 64, 64)
        
        x = self.conv2(x)           # (N, 64, 2, 32, 32)
        d = x
        
        x = self.conv_c(x)
        x = self.relu(x)
        x = self.conv_c1(x)
        x = x + d
        x = self.relu(x)        # (N, 64, 2, 32, 32)
  
        x = self.conv3(x)
        d1 = x                  #(N, 128, 1, 16, 16)
        
        x = self.conv_c2(x)
        x = self.relu(x)
        x = self.conv_c3(x)
        x = x + d1
        x = self.relu(x)        # (N, 128, 1, 16, 16)
        
        x = x.view(-1,128,16,16)
        x = self.head1(x)       #(N, 128, 16, 16)
        
        x = unsample1(x)        #(N, 128, 32, 32)
        
        x = self.conv_c4(x)
        x = self.relu(x)
        x = self.conv_c5(x)     #(N, 64, 32, 32)
        d1 = (self.maxpoolc(d)).view(-1, 64, 32, 32)
        d1 = self.head(d1)
        x = torch.cat((x,d1),dim=1)
        x = self.relu(x)          #(N, 128,  32, 32) 
        
        # x = x.view(-1, 64, 32, 32)
        
        
        # x = self.head(x)        # (N, 64, 32, 32)


        x = unsample1(x)                 # (N, 128, 64, 64)

        
        x = self.conv_d(x)
        x = self.relu(x)
        x = self.conv_d1(x)
        c = (self.maxpoolc(c)).view(-1, 32, 64, 64)
        x = torch.cat((x,c),dim=1)
        x = self.relu(x)          #(N, 64,  64, 64)  
        
        x = unsample1(x)         #(N, 64,  128, 128) 
        
        x = self.conv_e(x)
        x = self.relu(x)
        x = self.conv_e1(x)
        b = (self.maxpool(b)).view(-1, 16, 128, 128)
        x = torch.cat((x,b),dim=1)
        x = self.relu(x)          #(N, 32, 128, 128) 
        
        # x = self.convk(x)
        # x = self.relu(x)        #(N, 32, 128, 128)
        
        b = x
        x = self.conv6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = torch.cat((x,b),dim=1)
        x = self.relu(x)          #(N, 64, 128, 128) 
        
        x = self.conv8(x)       #(N, 32, 128, 128) 
        x = self.conv9(x)       #(N, 16, 128, 128)
        x = self.sigm(x)        #(N, 3, 128, 128)
        return x 



class BackboneNet_MAX_V5_2(nn.Module):
    def __init__(self):
        super(BackboneNet_MAX_V5_2, self).__init__()
        self.conv0 = nn.Conv3d(2, 16, (1, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv1 = nn.Conv3d(16, 32, (4, 4, 4), stride=2, padding=(1, 1, 1))

        self.conv_a = nn.Conv3d(16, 16, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_a1 = nn.Conv3d(16, 16, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_a2 = nn.Conv3d(16, 16, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_b = nn.Conv3d(32, 32, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_b1 = nn.Conv3d(32, 32, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_b2 = nn.Conv3d(32, 32, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_c = nn.Conv3d(64, 64, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_c1 = nn.Conv3d(64, 64, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_c11 = nn.Conv3d(64, 64, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_c2 = nn.Conv3d(128, 128, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_c3 = nn.Conv3d(128, 128, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_c33 = nn.Conv3d(128, 128, (1, 3, 3), stride=1, padding=(0, 1, 1))        
        self.conv_c4 = nn.Conv2d(128, 128, ( 3, 3), stride=1, padding=( 1, 1))
        self.conv_c5 = nn.Conv2d(128, 64, ( 3, 3), stride=1, padding=( 1, 1))         
        self.conv_c55 = nn.Conv2d(64, 64, ( 3, 3), stride=1, padding=( 1, 1))
        self.conv_d = nn.Conv2d(64, 64, ( 3, 3), stride=1, padding=( 1, 1))
        self.conv_d1 = nn.Conv2d(64, 32, ( 3, 3), stride=1, padding=( 1, 1))
        self.conv_d11 = nn.Conv2d(32, 32, ( 3, 3), stride=1, padding=( 1, 1))
        self.conv_e = nn.Conv2d(32, 32, (3, 3), stride=1, padding=( 1, 1))
        self.conv_e1 = nn.Conv2d(32, 32, ( 3, 3), stride=1, padding=( 1, 1))
        self.conv_e11 = nn.Conv2d(32, 16, ( 3, 3), stride=1, padding=( 1, 1))       
        self.conv2 = nn.Conv3d(32, 64, (2, 4, 4), stride=2, padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(64, 128, (2, 4, 4), stride=2, padding=(0, 1, 1))
        
        self.relu = nn.PReLU()


        # self.deconv2 = F.interpolate(size=(2,64,64), scale_factor=2, mode='bilinear')
        # self.deconv3 = F.interpolate(input,size=(4,128,128), scale_factor=2, mode='bilinear')


        # self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1))
        self.maxpoolc = nn.MaxPool3d(kernel_size=(2, 1, 1))

        self.head = DAHead_backbone64(in_channels=64,nclass=64)
        self.head1 = DAHead_backbone128(in_channels=128,nclass=128)
        self.conv6 = nn.Conv2d(32,32,( 3, 3), stride=1, padding=( 1, 1))
        self.conv7 = nn.Conv2d(32,32,( 3, 3), stride=1, padding=( 1, 1))
        self.conv8 = nn.Conv2d(32,16,( 3, 3), stride=1, padding=( 1, 1))
        self.conv9 = nn.Conv2d(16,3,( 1, 1), stride=1, padding=( 0, 0))
        self.convk = nn.Conv2d(16,32,( 3, 3), stride=1, padding=( 1, 1))
        self.sigm = nn.Sigmoid()
    def forward(self, x):
        # batch_size, in_chirps, n_channels, w, h = x.shape
        
        x = self.conv0(x)
        b = x
        x = self.relu(x) 
        
        x = self.conv_a(x)         #    (N, 16, 4, 128, 128)
        x = self.relu(x)
        x = self.conv_a1(x)
        x = self.relu(x)
        x = self.conv_a2(x)
        x = x + b                    #res 
        x = self.relu(x)            #(N, 16, 4, 128, 128)
        
        x = self.conv1(x)           # (N, 32, 2, 64, 64)
        c = x
        
        x = self.conv_b(x)
        x = self.relu(x)
        x = self.conv_b1(x)
        x = self.relu(x)
        x = self.conv_b2(x)
        x = x + c
        x = self.relu(x)        # (N, 32, 2, 64, 64)
        
        x = self.conv2(x)           # (N, 64, 2, 32, 32)
        d = x
        
        x = self.conv_c(x)
        x = self.relu(x)
        x = self.conv_c1(x)
        x = self.relu(x)
        x = self.conv_c11(x)
        x = x + d
        x = self.relu(x)        # (N, 64, 2, 32, 32)
  
        x = self.conv3(x)
        d1 = x                  #(N, 128, 1, 16, 16)
        
        x = self.conv_c2(x)
        x = self.relu(x)
        x = self.conv_c3(x)
        x = self.relu(x)
        x = self.conv_c33(x)
        x = x + d1
        x = self.relu(x)        # (N, 128, 1, 16, 16)
        
        x = x.view(-1,128,16,16)
        x = self.head1(x)       #(N, 128, 16, 16)
        
        x = unsample1(x)        #(N, 128, 32, 32)
        
        x = self.conv_c4(x)
        x = self.relu(x)
        x = self.conv_c5(x)     #(N, 64, 32, 32)
        x = self.relu(x)
        x = self.conv_c55(x)
        d1 = (self.maxpoolc(d)).view(-1, 64, 32, 32)
        d1 = self.head(d1)
        x = x + d1
        x = self.relu(x)          #(N, 64,  32, 32) 
        
        # x = x.view(-1, 64, 32, 32)
        
        
        # x = self.head(x)        # (N, 64, 32, 32)


        x = unsample1(x)                 # (N, 64, 64, 64)

        
        x = self.conv_d(x)
        x = self.relu(x)
        x = self.conv_d1(x)
        x = self.relu(x)
        x = self.conv_d11(x)
        c = (self.maxpoolc(c)).view(-1, 32, 64, 64)
        x = x + c
        x = self.relu(x)          #(N, 32,  64, 64)  
        
        x = unsample1(x)         #(N, 32,  128, 128) 
        
        x = self.conv_e(x)
        x = self.relu(x)
        x = self.conv_e1(x)
        x = self.relu(x)
        x = self.conv_e11(x)
        b = (self.maxpool(b)).view(-1, 16, 128, 128)
        x = x + b
        x = self.relu(x)          #(N, 16, 128, 128) 
        
        x = self.convk(x)
        x = self.relu(x)        #(N, 32, 128, 128)
        
        b = x
        x = self.conv6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = x + b
        x = self.relu(x)          #(N, 32, 128, 128) 
        
        x = self.conv8(x)       #(N, 32, 128, 128) 
        x = self.conv9(x)       #(N, 16, 128, 128)
        x = self.sigm(x)        #(N, 3, 128, 128)
        return x     


class BackboneNet_MAX_V2_1(nn.Module):
    def __init__(self):
        super(BackboneNet_MAX_V2_1, self).__init__()
        self.conv0 = nn.Conv3d(2, 16, (1, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv1 = nn.Conv3d(16, 32, (6, 4, 4), stride=2, padding=(2, 1, 1))

        self.conv_a = nn.Conv3d(16, 16, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_a1 = nn.Conv3d(16, 16, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_b = nn.Conv3d(32, 32, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_b1 = nn.Conv3d(32, 32, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_c = nn.Conv3d(64, 64, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_c1 = nn.Conv3d(64, 64, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_d = nn.Conv3d(64, 64, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_d1 = nn.Conv3d(64, 32, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_e = nn.Conv3d(32, 32, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_e1 = nn.Conv3d(32, 16, (1, 3, 3), stride=1, padding=(0, 1, 1))
               
        self.conv2 = nn.Conv3d(32, 64, (6, 4, 4), stride=2, padding=(2, 1, 1))
        self.conv_f = nn.Conv3d(16, 16, (2, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_f1 = nn.Conv3d(16, 3, (3, 3, 3), stride=1, padding=(0, 1, 1))
        self.relu = nn.PReLU()
        self.head = DAHead_backbone64(in_channels=64,nclass=64)
        self.sigmod = nn.Sigmoid()
    def forward(self, x):
        # batch_size, in_chirps, n_channels, w, h = x.shape
        
        x = self.conv0(x)
        b = x
        x = self.relu(x) 
        
        x = self.conv_a(x)         #    (N, 16, 4, 128, 128)
        x = self.relu(x)
        x = self.conv_a1(x)
        x = x + b                    #res 
        x = self.relu(x)            #(N, 16, 4, 128, 128)
        
        x = self.conv1(x)           # (N, 32, 2, 64, 64)
        c = x
        
        x = self.conv_b(x)
        x = self.relu(x)
        x = self.conv_b1(x)
        x = x + c
        x = self.relu(x)        # (N, 32, 2, 64, 64)
        
        x = self.conv2(x)           # (N, 64, 1, 32, 32)
        d = x
        
        x = self.conv_c(x)
        x = self.relu(x)
        x = self.conv_c1(x)
        x = x + d
        x = self.relu(x)        # (N, 64, 1, 32, 32)
  
        x = x.view(-1, 64, 32, 32)
        x = self.head(x)        # (N, 64, 32, 32)
        x = x.view(-1, 64,1, 32, 32)  # (N, 64, 1, 32, 32)


        # x = self.deconv2(x)         # (N, 64, 2, 64, 64)
        x = unsample(x)                 # (N, 64, 64, 64)

        
        x = self.conv_d(x)
        x = self.relu(x)
        x = self.conv_d1(x)
        x = x + c
        x = self.relu(x)          #(N, 32, 2, 64, 64)  
        
        x = unsample(x)         #(N, 32, 4, 128, 128) 
        
        x = self.conv_e(x)
        x = self.relu(x)
        x = self.conv_e1(x)
        x = x + b
        x = self.relu(x)          #(N, 16,4, 128, 128) 
        
        x = self.conv_f(x)
        x = self.conv_f1(x)
        # x = self.sigmoid(x)
        x = x.view(-1, 3, 128, 128)
        x = self.sigmod(x)
        return x 


class Conv2d_batchnorm(nn.Module):
    
    def __init__(self, num_in_filters, num_out_filters, kernel_size, stride = (1,1),padding = (1,1),activation = 'relu'):
        super(Conv2d_batchnorm, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters, kernel_size=kernel_size, stride=stride, padding = padding)
        # self.batchnorm = nn.BatchNorm2d(num_out_filters)
        self.relu = nn.PReLU()
    def forward(self,x):
        x = self.conv1(x)
        # x = self.batchnorm(x)

        if self.activation == 'relu':
            x = self.relu(x)
            return x
        else:
            return x


class Multiresblock2D(nn.Module):

    def __init__(self, num_in_channels):

        super(Multiresblock2D, self).__init__()


        out = num_in_channels//2
        self.shortcut = Conv2d_batchnorm(num_in_channels ,num_in_channels , kernel_size = (1,1), padding = (0,0),activation='None')

        self.conv_3x3 = Conv2d_batchnorm(num_in_channels, num_in_channels, kernel_size = (3,3), activation='relu')

        self.conv_5x5 = Conv2d_batchnorm(num_in_channels, num_in_channels, kernel_size = (3,3), activation='relu')

        self.conv_7x7 = Conv2d_batchnorm(num_in_channels, num_in_channels, kernel_size = (3,3), activation='relu')

        # self.batch_norm1 = nn.BatchNorm2d(num_in_channels)
        # self.batch_norm2 = nn.BatchNorm2d(num_in_channels)
        self.relu = nn.PReLU()
        self.conv1 = nn.Conv2d(num_in_channels, out, (1, 1), stride=1, padding=(0, 0))
    def forward(self,x):

        shrtct = self.shortcut(x)

        a = self.conv_3x3(x)
        b = self.conv_5x5(a)
        c = self.conv_7x7(b)

        x = a+b+c   #torch.cat([a,b,c],axis=1)
        # x = self.batch_norm1(x)

        x = x + shrtct
        # x = self.batch_norm2(x)
        
        x = self.conv1(x)
        x = self.relu(x)
        return x

class Multiresblock2D1(nn.Module):
    
    def __init__(self, num_in_channels):

        super(Multiresblock2D1, self).__init__()


        out = num_in_channels//4
        self.shortcut = Conv2d_batchnorm(num_in_channels ,num_in_channels , kernel_size = (1,1), padding=(0,0),activation='None')

        self.conv_3x3 = Conv2d_batchnorm(num_in_channels, num_in_channels, kernel_size = (3,3), activation='relu')

        self.conv_5x5 = Conv2d_batchnorm(num_in_channels, num_in_channels, kernel_size = (3,3), activation='relu')

        self.conv_7x7 = Conv2d_batchnorm(num_in_channels, num_in_channels, kernel_size = (3,3), activation='relu')

        # self.batch_norm1 = nn.BatchNorm2d(num_in_channels)
        # self.batch_norm2 = nn.BatchNorm2d(num_in_channels)
        self.relu = nn.PReLU()
        self.conv1 = nn.Conv2d(num_in_channels, out, (1, 1), stride=1, padding=(0, 0))
    def forward(self,x):

        shrtct = self.shortcut(x)

        a = self.conv_3x3(x)
        b = self.conv_5x5(a)
        c = self.conv_7x7(b)
        
        x = a+b+c#torch.cat([a,b,c],axis=1)
        # x = self.batch_norm1(x)

        x = x + shrtct
        # x = self.batch_norm2(x)
        
        x = self.conv1(x)
        x = self.relu(x)
        return x

class Conv3d_batchnorm(nn.Module):

    def __init__(self, num_in_filters, num_out_filters, kernel_size, stride = (1,1,1), padding = (1,1,1), activation = 'relu'):
        super(Conv3d_batchnorm, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv3d(in_channels=num_in_filters, out_channels=num_out_filters, kernel_size=kernel_size, stride=stride, padding = padding)
        # self.batchnorm = nn.BatchNorm3d(num_out_filters)
        self.relu = nn.PReLU()
    def forward(self,x):
        x = self.conv1(x)
        # x = self.batchnorm(x)
        
        if self.activation == 'relu':
            x = self.relu(x)
            return x
        else:
            return x


class Multiresblock(nn.Module):

    def __init__(self, num_in_channels):

        super(Multiresblock, self).__init__()



        self.shortcut = Conv3d_batchnorm(num_in_channels ,num_in_channels , kernel_size = (1,1,1),padding = (0,0,0), activation='None')

        self.conv_3x3 = Conv3d_batchnorm(num_in_channels, num_in_channels, kernel_size = (3,3,3), activation='relu')

        self.conv_5x5 = Conv3d_batchnorm(num_in_channels, num_in_channels, kernel_size = (3,3,3), activation='relu')

        self.conv_7x7 = Conv3d_batchnorm(num_in_channels, num_in_channels, kernel_size = (3,3,3), activation='relu')

        # self.batch_norm1 = nn.BatchNorm3d(num_in_channels)
        # self.batch_norm2 = nn.BatchNorm3d(num_in_channels)

        self.relu = nn.PReLU()
    def forward(self,x):

        res = self.shortcut(x)

        a = self.conv_3x3(x)
        b = self.conv_5x5(a)
        c = self.conv_7x7(b)

        x = a+b+c           #torch.cat([a,b,c],axis=1)
        # x = self.batch_norm1(x)

        x = x + res
        # x = self.batch_norm2(x)
        x = self.relu(x)

        return x

class MultiResUNet0(nn.Module):
    def __init__(self):
        super(MultiResUNet0, self).__init__()
        self.conv0 = nn.Conv3d(2, 16, (1, 1, 1), stride=1, padding=(0, 0, 0))
        self.en_block1 = Multiresblock(16)
        self.conv1 = nn.Conv3d(16, 32, (4, 4, 4), stride=2, padding=(1, 1, 1))
        self.en_block2 = Multiresblock(32)
        self.conv2 = nn.Conv3d(32, 64, (6, 4, 4), stride=2, padding=(2, 1, 1))
        self.en_block3 = Multiresblock(64)
        self.head = DAHead_backbone64(in_channels=64,nclass=64)
        self.de_block1 = Multiresblock2D(64)
        self.de_block2 = Multiresblock2D1(64)
        self.relu = nn.PReLU()

        # self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1))
        self.maxpoolc = nn.MaxPool3d(kernel_size=(2, 1, 1))

        
        # self.conv8 = nn.Conv2d(64,16,( 3, 3), stride=1, padding=( 1, 1))
        self.conv9 = nn.Conv2d(32,3,( 1, 1), stride=1, padding=( 0, 0))
        # self.convk = nn.Conv2d(16,32,( 3, 3), stride=1, padding=( 1, 1))
        self.sigm = nn.Sigmoid()
    def forward(self, x):
        # batch_size, in_chirps, n_channels, w, h = x.shape
        
        x = self.conv0(x)
        x = self.en_block1(x)       #(N, 16, 4, 128, 128)
        b = x
        x = self.conv1(x)           # (N, 32, 2, 64, 64)
        x = self.en_block2(x)       # (N, 32, 2, 64, 64)
        c = x
        x = self.conv2(x)           # (N, 64, 1, 32, 32)
        x = self.en_block3(x)       # (N, 64, 1, 32, 32)
  
        x = x.view(-1, 64, 32, 32)
        x = self.head(x)        # (N, 64, 32, 32)


        x = unsample1(x)                 # (N, 64, 64, 64)

        
        x = self.de_block1(x)
        c = (self.maxpoolc(c)).view(-1, 32, 64, 64)
        x = torch.cat((x,c),dim=1)
        x = self.relu(x)          #(N, 64,  64, 64)  
        
        x = unsample1(x)         #(N, 64,  128, 128) 
        
        x = self.de_block2(x)
        b = (self.maxpool(b)).view(-1, 16, 128, 128)
        x = torch.cat((x,b),dim=1)
        # x = self.relu(x)          #(N, 32, 128, 128) 
        
        # x = self.conv8(x)       #(N, 32, 128, 128) 
        x = self.conv9(x)       #(N, 16, 128, 128)
        x = self.sigm(x)        #(N, 3, 128, 128)
        return x 
##############################################################################################################
class Multiresblock2D_1(nn.Module):

    def __init__(self, num_in_channels,in_channel,out_1):

        super(Multiresblock2D_1, self).__init__()


        out = num_in_channels*3
        self.shortcut = Conv2d_batchnorm(in_channel ,out , kernel_size = (1,1), padding = (0,0),activation='None')

        self.conv_3x3 = Conv2d_batchnorm(in_channel, num_in_channels, kernel_size = (3,3), activation='relu')

        self.conv_5x5 = Conv2d_batchnorm(num_in_channels, num_in_channels, kernel_size = (3,3), activation='relu')

        self.conv_7x7 = Conv2d_batchnorm(num_in_channels, num_in_channels, kernel_size = (3,3), activation='relu')
        # self.att = DAHead_Channel(out)
        # self.batch_norm1 = nn.BatchNorm2d(out)
        # self.batch_norm2 = nn.BatchNorm2d(out)
        self.relu = nn.PReLU()
        self.conv1 = nn.Conv2d(out, out_1, (1, 1), stride=1, padding=(0, 0))
    def forward(self,x):

        shrtct = self.shortcut(x)

        a = self.conv_3x3(x)
        b = self.relu(a)
        b = self.conv_5x5(b)
        c = self.relu(b)
        c = self.conv_7x7(c)

        x = torch.cat([a,b,c],axis=1)
        # x = self.batch_norm1(x)
        # x = self.att(x)
        x = x + shrtct
        # x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.conv1(x)
        
        return x

class Multiresblock2D1_1(nn.Module):
    
    def __init__(self, num_in_channels,in_channel):

        super(Multiresblock2D1_1, self).__init__()


        out = num_in_channels*3
        self.shortcut = Conv2d_batchnorm(in_channel ,out , kernel_size = (1,1), padding=(0,0),activation='None')

        self.conv_3x3 = Conv2d_batchnorm(in_channel, num_in_channels, kernel_size = (3,3), activation='relu')

        self.conv_5x5 = Conv2d_batchnorm(num_in_channels, num_in_channels, kernel_size = (3,3), activation='relu')

        self.conv_7x7 = Conv2d_batchnorm(num_in_channels, num_in_channels, kernel_size = (3,3), activation='relu')
        # self.att = DAHead_Channel(out)
        # self.batch_norm1 = nn.BatchNorm2d(out)
        # self.batch_norm2 = nn.BatchNorm2d(out)
        self.relu = nn.PReLU()
        self.conv1 = nn.Conv2d(out, num_in_channels, (1, 1), stride=1, padding=(0, 0))
    def forward(self,x):

        shrtct = self.shortcut(x)

        a = self.conv_3x3(x)
        b = self.relu(a)
        b = self.conv_5x5(b)
        c = self.relu(b)
        c = self.conv_7x7(c)

        x = torch.cat([a,b,c],axis=1)
        # x = self.batch_norm1(x)
        # x = self.att(x)
        x = x + shrtct
        # x = self.batch_norm2(x)
        x = self.relu(x)        
        x = self.conv1(x)

        return x


class Multiresblock_1(nn.Module):

    def __init__(self, num_in_channels):

        super(Multiresblock_1, self).__init__()


        out = num_in_channels*3
        self.shortcut = Conv3d_batchnorm(num_in_channels ,out , kernel_size = (1,1,1),padding = (0,0,0), activation='None')

        self.conv_3x3 = Conv3d_batchnorm(num_in_channels, num_in_channels, kernel_size = (3,3,3), activation='relu')

        self.conv_5x5 = Conv3d_batchnorm(num_in_channels, num_in_channels, kernel_size = (3,3,3), activation='relu')

        self.conv_7x7 = Conv3d_batchnorm(num_in_channels, num_in_channels, kernel_size = (3,3,3), activation='relu')
        # self.att = DAHead_Channel(out)
        # self.batch_norm1 = nn.BatchNorm3d(out)
        # self.batch_norm2 = nn.BatchNorm3d(out)

        self.relu = nn.PReLU()
    def forward(self,x):

        res = self.shortcut(x)

        a = self.conv_3x3(x)
        b = self.relu(a)
        b = self.conv_5x5(b)
        c = self.relu(b)
        c = self.conv_7x7(c)

        x = torch.cat([a,b,c],axis=1)
        # x = self.batch_norm1(x)
        # x = self.att(x)
        x = x + res
        # x = self.batch_norm2(x)
        x = self.relu(x)

        return x

class MultiResUNet1(nn.Module):
    def __init__(self):
        super(MultiResUNet1, self).__init__()
        self.conv0 = nn.Conv3d(2, 16, (1, 1, 1), stride=1, padding=(0, 0, 0))
        self.en_block1 = Multiresblock_1(16)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        # self.conv1 = nn.Conv3d(16, 32, (4, 4, 4), stride=2, padding=(1, 1, 1))
        self.conv1 = nn.Conv3d(48, 16, (1, 1, 1), stride=1, padding=(0, 0, 0))
        self.en_block2 = Multiresblock_1(48)
        self.conv2 = nn.Conv3d(144, 64, (1, 1, 1), stride=1, padding=(0, 0, 0))
        # self.conv2 = nn.Conv3d(32, 64, (6, 4, 4), stride=2, padding=(2, 1, 1))
        self.en_block3 = Multiresblock_1(64)
        self.head = DAHead_backbone192(in_channels=192)
        # self.head1 = DAHead_backbone64(in_channels=64,nclass=64)
        self.de_block1 = Multiresblock2D_1(32,192,32)
        self.de_block2 = Multiresblock2D1_1(16,64)
        self.relu = nn.PReLU()
        self.conv3 = nn.Conv3d(64, 32, (1, 1, 1), stride=1, padding=(0, 0, 0))
        # self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1))
        self.maxpoolc = nn.MaxPool3d(kernel_size=(2, 1, 1))

        
        # self.conv8 = nn.Conv2d(64,16,( 3, 3), stride=1, padding=( 1, 1))
        self.conv9 = nn.Conv2d(32,3,( 1, 1), stride=1, padding=( 0, 0))
        # self.convk = nn.Conv2d(16,32,( 3, 3), stride=1, padding=( 1, 1))
        self.sigm = nn.Sigmoid()
    def forward(self, x):
        # batch_size, in_chirps, n_channels, w, h = x.shape
        
        x = self.conv0(x)
        x = self.en_block1(x)       #(N, 48, 4, 128, 128)
        # x = self.conv1(x)           #(N, 48, 4, 128, 128)
        b = x
        x = self.pool1(x)           # (N, 48, 2, 64, 64)
        x = self.en_block2(x)       # (N, 144, 2, 64, 64)
        x= self.conv2(x)            # (N, 64, 2, 64, 64)
        c = x
        x = self.pool1(x)           # (N, 64, 1, 32, 32)
        x = self.en_block3(x)       # (N, 192, 1, 32, 32)

        x = x.view(-1, 192, 32, 32)
        x = self.head(x)        # (N, 192, 32, 32)


        x = unsample1(x)                 # (N, 192, 64, 64)

        
        x = self.de_block1(x)           ## (N, 32, 64, 64)
        c = (self.maxpoolc(self.conv3(c))).view(-1, 32, 64, 64)
        x = torch.cat((x,c),dim=1)
        # x = self.relu(x)          #(N, 64,  64, 64)  
        # x = self.head1(x)
        x = unsample1(x)         #(N, 64,  128, 128) 
        
        x = self.de_block2(x)
        b = (self.maxpool(self.conv1(b))).view(-1, 16, 128, 128)
        x = torch.cat((x,b),dim=1)
        # x = self.relu(x)          #(N, 32, 128, 128) 
        
        # x = self.conv8(x)       #(N, 32, 128, 128) 
        x = self.conv9(x)       #(N, 3, 128, 128)
        x = self.sigm(x)        #(N, 3, 128, 128)
        return x  

##########################################################
class Multiresblock2D_20(nn.Module):

    def __init__(self, num_in_channels):

        super(Multiresblock2D_20, self).__init__()


        # out = num_in_channels*3
        self.shortcut = Conv2d_batchnorm(512 ,640 , kernel_size = (1,1), padding = (0,0),activation='None')

        self.conv_3x3 = Conv2d_batchnorm(512, 256, kernel_size = (3,3), activation='relu')

        self.conv_5x5 = Conv2d_batchnorm(256, 256, kernel_size = (3,3), activation='relu')

        self.conv_7x7 = Conv2d_batchnorm(256, 128, kernel_size = (3,3), activation='relu')

        # self.batch_norm1 = nn.BatchNorm2d(out)
        # self.batch_norm2 = nn.BatchNorm2d(out)
        self.relu = nn.PReLU()
        self.conv1 = nn.Conv2d(640, 256, (1, 1), stride=1, padding=(0, 0))
    def forward(self,x):

        shrtct = self.shortcut(x)

        a = self.conv_3x3(x)
        b = self.conv_5x5(a)
        c = self.conv_7x7(b)

        x = torch.cat([a,b,c],axis=1)
        # x = self.batch_norm1(x)

        x = x + shrtct
        # x = self.batch_norm2(x)
        
        x = self.conv1(x)
        x = self.relu(x)
        return x
class Multiresblock2D_21(nn.Module):

    def __init__(self, num_in_channels):

        super(Multiresblock2D_21, self).__init__()


        # out = num_in_channels*3
        self.shortcut = Conv2d_batchnorm(320 ,320 , kernel_size = (1,1), padding = (0,0),activation='None')

        self.conv_3x3 = Conv2d_batchnorm(320, 128, kernel_size = (3,3), activation='relu')

        self.conv_5x5 = Conv2d_batchnorm(128, 128, kernel_size = (3,3), activation='relu')

        self.conv_7x7 = Conv2d_batchnorm(128, 64, kernel_size = (3,3), activation='relu')

        # self.batch_norm1 = nn.BatchNorm2d(out)
        # self.batch_norm2 = nn.BatchNorm2d(out)
        self.relu = nn.PReLU()
        self.conv1 = nn.Conv2d(320, 128, (1, 1), stride=1, padding=(0, 0))
    def forward(self,x):

        shrtct = self.shortcut(x)

        a = self.conv_3x3(x)
        b = self.conv_5x5(a)
        c = self.conv_7x7(b)

        x = torch.cat([a,b,c],axis=1)
        # x = self.batch_norm1(x)

        x = x + shrtct
        # x = self.batch_norm2(x)
        x = self.conv1(x)
        x = self.relu(x)
        
        return x
class Multiresblock2D_22(nn.Module):

    def __init__(self, num_in_channels):

        super(Multiresblock2D_22, self).__init__()


        # out = num_in_channels*3
        self.shortcut = Conv2d_batchnorm(160 ,112 , kernel_size = (1,1), padding = (0,0),activation='None')

        self.conv_3x3 = Conv2d_batchnorm(160, 64, kernel_size = (3,3), activation='relu')

        self.conv_5x5 = Conv2d_batchnorm(64, 32, kernel_size = (3,3), activation='relu')

        self.conv_7x7 = Conv2d_batchnorm(32, 16, kernel_size = (3,3), activation='relu')

        # self.batch_norm1 = nn.BatchNorm2d(out)
        # self.batch_norm2 = nn.BatchNorm2d(out)
        self.relu = nn.PReLU()
        self.conv1 = nn.Conv2d(112, 32, (1, 1), stride=1, padding=(0, 0))
    def forward(self,x):

        shrtct = self.shortcut(x)

        a = self.conv_3x3(x)
        b = self.conv_5x5(a)
        c = self.conv_7x7(b)

        x = torch.cat([a,b,c],axis=1)
        # x = self.batch_norm1(x)

        x = x + shrtct
        # x = self.batch_norm2(x)
        x = self.conv1(x)
        x = self.relu(x)
        
        return x
class MultiResUNet2(nn.Module):
    def __init__(self):
        super(MultiResUNet2, self).__init__()
        self.conv0 = nn.Conv3d(2, 16, (1, 1, 1), stride=1, padding=(0, 0, 0))
        self.en_block1 = Multiresblock_1(16)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        # self.conv1 = nn.Conv3d(16, 32, (4, 4, 4), stride=2, padding=(1, 1, 1))
        self.conv1 = nn.Conv3d(48, 16, (1, 1, 1), stride=1, padding=(0, 0, 0))
        self.en_block2 = Multiresblock_1(48)
        self.conv2 = nn.Conv3d(144, 64, (1, 1, 1), stride=1, padding=(0, 0, 0))
        # self.conv2 = nn.Conv3d(32, 64, (6, 4, 4), stride=2, padding=(2, 1, 1))
        self.en_block3 = Multiresblock_1(64)
        self.conv3 = nn.Conv3d(192, 128, (1, 1, 1), stride=1, padding=(0, 0, 0))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.en_block4 = Multiresblock_1(128)
        self.head = DAHead_backbone384(in_channels=384,nclass=384)
        self.de_block3 = Multiresblock2D_20(256)
        self.de_block1 = Multiresblock2D_21(32)
        self.conv7 = nn.Conv3d(48, 48, (3, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv8 = nn.Conv3d(48, 32, (2, 3, 3), stride=1, padding=(0, 1, 1))
        self.de_block2 = Multiresblock2D_22(16)
        self.conv4 = nn.Conv3d(128,128,(1,3,3),(1,1,1),(0,1,1))
        self.relu = nn.PReLU()
        self.conv5 = nn.Conv3d(64, 64, (3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv6 = nn.Conv3d(64, 64, (2, 3, 3), stride=1, padding=(0, 1, 1))
        # self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1))
        self.maxpoolc = nn.MaxPool3d(kernel_size=(2, 1, 1))

        
        # self.conv8 = nn.Conv2d(64,16,( 3, 3), stride=1, padding=( 1, 1))
        self.conv9 = nn.Conv2d(32,3,( 1, 1), stride=1, padding=( 0, 0))
        # self.convk = nn.Conv2d(16,32,( 3, 3), stride=1, padding=( 1, 1))
        self.sigm = nn.Sigmoid()
    def forward(self, x):
        # batch_size, in_chirps, n_channels, w, h = x.shape
        
        x = self.conv0(x)
        x = self.en_block1(x)       #(N, 48, 4, 128, 128)
        # x = self.conv1(x)           #(N, 48, 4, 128, 128)
        b = x
        x = self.pool1(x)           # (N, 48, 2, 64, 64)
        x = self.en_block2(x)       # (N, 144, 2, 64, 64)
        x= self.conv2(x)            # (N, 64, 2, 64, 64)
        c = x
        x = self.pool1(x)           # (N, 64, 1, 32, 32)
        x = self.en_block3(x)       # (N, 192, 1, 32, 32)
        x = self.conv3(x)           # (N, 128, 1, 32, 32)
        d = x
        x = self.pool2(x)           # (N, 128, 1, 16, 16)
        x = self.en_block4(x)       # (N, 384, 1, 16, 16)
        x = x.view(-1, 384, 16, 16)
        x = self.head(x)        # (N, 384, 16, 16)
        x = unsample1(x)                 # (N, 384, 32, 32)

        d = self.conv4(d).view(-1, 128, 32, 32)
        d = self.relu(d)
        x = torch.cat((x,d),dim=1)       #   (N, 512, 32, 32)    
        x = self.de_block3(x)           ## (N, 256, 32, 32)

        # x = self.relu(x)          #(N, 192,  32, 32)  

        x = unsample1(x)                 # (N, 256, 64, 64)

        c = self.conv6(self.conv5(c)).view(-1, 64, 64, 64)#(N, 64, 64, 64)
        c = self.relu(c)
        x = torch.cat((x,c),dim=1)        #(N, 320, 64, 64)
        x = self.de_block1(x)           ## (N, 128, 64, 64)

        # x = self.relu(x)          #(N, 64,  64, 64)  
        
        x = unsample1(x)         #(N, 128,  128, 128) 
        
        b = self.conv8(self.conv7(b)).view(-1, 32, 128, 128)
        b = self.relu(b)
        x = torch.cat((x,b),dim=1)      #(N, 160,  128, 128)
        x = self.de_block2(x)       #(N, 32,  128, 128)
        
        # x = self.relu(x)          #(N, 32, 128, 128) 
        
        # x = self.conv8(x)       #(N, 32, 128, 128) 
        x = self.conv9(x)       #(N, 3, 128, 128)
        x = self.sigm(x)        #(N, 3, 128, 128)
        return x  

##############################################################################################################
class Multiresblock_conv(torch.nn.Module):

    def __init__(self, num_in_channels, num_filters):

        super(Multiresblock_conv, self).__init__()
        # self.alpha = alpha
        self.W = num_filters 

        filt_cnt_3x3 = round(self.W*0.167)
        filt_cnt_5x5 = round(self.W*0.333)
        filt_cnt_7x7 = round(self.W*0.5)
        num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7

        self.shortcut = Conv2d_batchnorm(num_in_channels ,num_out_filters , kernel_size = (1,1),padding=(0,0), activation='None')

        self.conv_3x3 = Conv2d_batchnorm(num_in_channels, filt_cnt_3x3, kernel_size = (3,3), activation='relu')

        self.conv_5x5 = Conv2d_batchnorm(filt_cnt_3x3, filt_cnt_5x5, kernel_size = (3,3), activation='relu')

        self.conv_7x7 = Conv2d_batchnorm(filt_cnt_5x5, filt_cnt_7x7, kernel_size = (3,3), activation='relu')
        self.relu = nn.PReLU()

    def forward(self,x):

        shrtct = self.shortcut(x)
        
        a = self.conv_3x3(x)
        b = self.conv_5x5(a)
        c = self.conv_7x7(b)

        x = torch.cat([a,b,c],axis=1)


        x = x + shrtct

        # x = torch.nn.functional.relu(x)

        return x

class Respath(nn.Module):

    def __init__(self, num_in_filters, num_out_filters, respath_length):

        super(Respath, self).__init__()

        self.respath_length = respath_length
        self.shortcuts = nn.ModuleList([])
        self.convs = nn.ModuleList([])
        # self.bns = torch.nn.ModuleList([])

        for i in range(self.respath_length):
            if(i==0):
                self.shortcuts.append(Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (1,1), padding=0,activation='None'))
                self.convs.append(Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (3,3),activation='relu'))

                
            else:
                self.shortcuts.append(Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (1,1),padding=0, activation='None'))
                self.convs.append(Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (3,3), activation='relu'))

            # self.bns.append(torch.nn.BatchNorm2d(num_out_filters))
		
	
    def forward(self,x):

        for i in range(self.respath_length):

            shortcut = self.shortcuts[i](x)

            x = self.convs[i](x)
            # x = self.bns[i](x)
            

            x = x + shortcut
            
            

        return x
class Respath_3D(nn.Module):

    def __init__(self,in_channel,out_channel, respath_length):

        super(Respath_3D, self).__init__()

        self.respath_length = respath_length
        
        self.convs1 = nn.Conv3d(in_channel,out_channel,(2,3,3),(2,1,1),(0,1,1))
        # self.bns = torch.nn.ModuleList([])

        
        if(self.respath_length==1):
            
            self.convs0 = nn.Conv3d(48,32,(3,3,3),(2,1,1),(1,1,1))
            self.shortcuts = nn.MaxPool3d((4,1,1))
            self.con1 = nn.Conv3d(48,16,1,1,0)
            self.relu = nn.PReLU()
            
        else:
            self.shortcuts = nn.MaxPool3d((2,1,1))
            self.con1 = nn.Conv3d(64,32,1,1,0)
            self.relu = nn.ReLU()

           
		
	
    def forward(self,x):

        if(self.respath_length==1):
            
            d = self.con1(x)
            d = self.shortcuts(d)
            x = self.convs0(x)
            x = self.relu(x)
            x = self.convs1(x)
            x = x + d
            x = self.relu(x)
            
        else:
            d = self.con1(x)
            d = self.shortcuts(d)
            x = self.convs1(x)
            x = x + d
            x = self.relu(x)

            
            

        return x
##########################################################
class Multiresblock_3D_2D(nn.Module):

    def __init__(self, num_in_channels,in_channel):

        super(Multiresblock_3D_2D, self).__init__()


        out = num_in_channels*2
        self.shortcut = Conv3d_batchnorm(in_channel ,out , kernel_size = (1,1,1),padding = (0,0,0), activation='None')

        self.conv_3x3 = Conv3d_batchnorm(in_channel, num_in_channels, kernel_size = (3,1,1), padding = (1,0,0),activation='relu')

        self.conv_5x5 = Conv3d_batchnorm(num_in_channels, num_in_channels, kernel_size = (3,1,1),padding = (1,0,0), activation='relu')

        # self.conv_7x7 = Conv3d_batchnorm(num_in_channels, num_in_channels, kernel_size = (3,1,1), activation='relu')
        self.convpool = Conv3d_batchnorm(out, out, kernel_size = (3,1,1),stride=(2,1,1),padding = (1,0,0), activation='relu')
        # self.convpool1 = Conv3d_batchnorm(48, 32, kernel_size = (2,1,1),padding = (0,0,0), activation='relu')
        # self.batch_norm1 = nn.BatchNorm3d(out)
        # self.batch_norm2 = nn.BatchNorm3d(out)

        # self.relu = nn.PReLU()
    def forward(self,x):

        res = self.shortcut(x)

        a = self.conv_3x3(x)
        b = self.conv_5x5(a)
        # c = self.conv_7x7(b)

        x = torch.cat([a,b],axis=1)
        # x = self.batch_norm1(x)

        x = x + res
        
        x = self.convpool(x)
        # x = self.convpool1(x)

        return x



class MultiResUNet_2D(nn.Module):
    def __init__(self, alpha=1):
        super(MultiResUNet_2D, self).__init__()
        # self.conv0 = nn.Conv3d(2, 16, (1, 1, 1), stride=1, padding=(0, 0, 0))
        
        self.alpha = alpha
        self.en_block1 = Multiresblock_3D_2D(16,2)    
        self.en_block11 = Multiresblock_3D_2D(32,32)  
        self.en_1 = nn.Conv2d(64, 32, 1,1,0)      
        self.in_filters1 = round(32*self.alpha*0.167)+round(32*self.alpha*0.333)+round(32*self.alpha* 0.5)
        self.pool1 = nn.Conv2d(32,self.in_filters1,3,2,1)
        self.respath1 = Respath(self.in_filters1,32,respath_length=4)

        self.en_block2 = Multiresblock_conv(self.in_filters1,32*2)
        self.in_filters2 = round(32*2*self.alpha*0.167)+round(32*2*self.alpha*0.333)+round(32*2*self.alpha* 0.5)
        self.pool2 =  nn.Conv2d(64,64,3,2,1) 
        self.respath2 = Respath(self.in_filters2,32*2,respath_length=3)

        self.en_block3 =  Multiresblock_conv(self.in_filters2,32*4)
        self.in_filters3 = round(32*4*self.alpha*0.167)+round(32*4*self.alpha*0.333)+round(32*4*self.alpha* 0.5)
        self.pool3 =  nn.Conv2d(128,128,3,2,1)
        self.respath3 = Respath(self.in_filters3,32*4,respath_length=2)
        
        self.en_block4 = Multiresblock_conv(self.in_filters3,32*8)
        self.in_filters4 = round(32*8*self.alpha*0.167)+round(32*8*self.alpha*0.333)+round(32*8*self.alpha* 0.5)
        self.head = DAHead_backbone256(in_channels=256,nclass=256)
        # self.respath4 = Respath(self.in_filters4,32*8,respath_length=1)
       
        self.in_filters6 = round(32*8*self.alpha*0.167)+round(32*8*self.alpha*0.333)+round(32*8*self.alpha* 0.5)

         
        self.concat_filters2 = 384
        self.de_block1 = Multiresblock_conv(self.concat_filters2,32*4)
        self.in_filters7 = round(32*4*self.alpha*0.167)+round(32*4*self.alpha*0.333)+round(32*4*self.alpha* 0.5)

        # self.upsample8 = torch.nn.ConvTranspose2d(self.in_filters7,32*2,kernel_size=(2,2),stride=(2,2))
        self.concat_filters3 = 192
        self.de_block2 = Multiresblock_conv(self.concat_filters3,32*2)
        self.in_filters8 = round(32*2*self.alpha*0.167)+round(32*2*self.alpha*0.333)+round(32*2*self.alpha* 0.5)

        # self.upsample9 = torch.nn.ConvTranspose2d(self.in_filters8,32,kernel_size=(2,2),stride=(2,2))
        self.concat_filters4 = 96
        self.de_block3 = Multiresblock_conv(self.concat_filters4,32)
        self.in_filters9 = round(32*self.alpha*0.167)+round(32*self.alpha*0.333)+round(32*self.alpha* 0.5)

        self.conv_final = Conv2d_batchnorm(self.in_filters9, 4, kernel_size = (1,1), padding=0,activation='None')

        self.relu = nn.PReLU()
        self.sigm = nn.Sigmoid()
    def forward(self, x):
        # batch_size, in_chirps, n_channels, w, h = x.shape
        
        
        x = self.en_block1(x)       #(N, 32, 128, 128)
        x = self.en_block11(x).view(-1, 64, 128, 128)
        x = self.en_1(x)            #(N, 32, 128, 128)
        b = x
        
        x = self.pool1(x)           # (N, 32,  64, 64)
        x = self.relu(x)
        x = self.en_block2(x)       # (N, 64,  64, 64)
        c = x
        
        x = self.pool2(x)           # (N, 64,  32, 32) 
        x = self.relu(x)
        x = self.en_block3(x)       # (N, 128,  32, 32)
        d = x

        x = self.pool3(x)           # (N, 128,  16, 16)
        x = self.relu(x)
        x = self.en_block4(x)       # (N, 256,  16, 16)
        
        x = self.head(x)        # (N, 256, 16, 16)
        x = unsample1(x)                 # (N, 256, 32, 32)

        d = self.respath3(d)             #(N, 128, 32, 32)
        x = torch.cat((x,d),dim=1)       #   (N, 384, 32, 32)    
        x = self.de_block1(x)           ## (N, 128, 32, 32)

        x = unsample1(x)                 # (N, 128, 64, 64)

        c = self.respath2(c)            #(N, 64, 64, 64)        
        x = torch.cat((x,c),dim=1)        #(N, 192, 64, 64)
        x = self.de_block2(x)           ## (N, 64, 64, 64)

        # x = self.relu(x)          #(N, 64,  64, 64)  
        
        x = unsample1(x)         #(N, 64,  128, 128) 
        
        b = self.respath1(b)        #(N, 32,  128, 128) 
        
        x = torch.cat((x,b),dim=1)      #(N, 96,  128, 128)
        x = self.de_block3(x)       #(N, 32,  128, 128)
        
        # x = self.relu(x)          #(N, 32, 128, 128) 
        
        # x = self.conv8(x)       #(N, 32, 128, 128) 
        x = self.conv_final(x)       #(N, 3, 128, 128)
        x = self.sigm(x)        #(N, 3, 128, 128)
        return x  

class MultiResUNet1_1(nn.Module):
    def __init__(self):
        super(MultiResUNet1_1, self).__init__()
        self.conv0 = nn.Conv3d(2, 16, (1, 1, 1), stride=1, padding=(0, 0, 0))
        self.en_block1 = Multiresblock_1(16)
        self.pool1 = nn.Conv3d(48,48,(3,3,3),2,1)
        self.pool2 = nn.Conv3d(64,64,(2,3,3),2,(0,1,1))
        # self.conv1 = nn.Conv3d(16, 32, (4, 4, 4), stride=2, padding=(1, 1, 1))
        # self.conv1 = nn.Conv3d(48, 16, (1, 1, 1), stride=1, padding=(0, 0, 0))
        self.en_block2 = Multiresblock_1(48)
        self.conv2 = nn.Conv3d(144, 64, (1, 1, 1), stride=1, padding=(0, 0, 0))
        # self.conv2 = nn.Conv3d(32, 64, (6, 4, 4), stride=2, padding=(2, 1, 1))
        self.en_block3 = Multiresblock_1(64)
        self.head = DAHead_backbone192(in_channels=192)
        # self.head1 = DAHead_backbone64(in_channels=64,nclass=64)
        self.de_block1 = Multiresblock2D_1(32,224,64)
        self.de_block2 = Multiresblock2D1_1(16,80)
        self.relu = nn.PReLU()
        self.conv3 = nn.Conv3d(64, 32, (1, 1, 1), stride=1, padding=(0, 0, 0))
        # self.relu = nn.LeakyReLU()
        self.maxpool = Respath_3D(32,16,1)
        self.maxpoolc = Respath_3D(64,32,0)

        
        # self.conv8 = nn.Conv2d(64,16,( 3, 3), stride=1, padding=( 1, 1))
        self.conv9 = nn.Conv2d(16,3,( 1, 1), stride=1, padding=( 0, 0))
        # self.convk = nn.Conv2d(16,32,( 3, 3), stride=1, padding=( 1, 1))
        self.sigm = nn.Sigmoid()
    def forward(self, x):
        # batch_size, in_chirps, n_channels, w, h = x.shape
        
        x = self.conv0(x)
        x = self.en_block1(x)       #(N, 48, 4, 128, 128)
        # x = self.conv1(x)           #(N, 48, 4, 128, 128)
        b = x
        x = self.pool1(x)           # (N, 48, 2, 64, 64)
        x = self.en_block2(x)       # (N, 144, 2, 64, 64)
        x= self.conv2(x)            # (N, 64, 2, 64, 64)
        c = x
        x = self.pool2(x)           # (N, 64, 1, 32, 32)
        x = self.en_block3(x)       # (N, 192, 1, 32, 32)

        x = x.view(-1, 192, 32, 32)
        x = self.head(x)        # (N, 192, 32, 32)


        x = unsample1(x)                 # (N, 192, 64, 64)

        c = (self.maxpoolc(c)).view(-1, 32, 64, 64)
        x = torch.cat((x,c),dim=1)        
        x = self.de_block1(x)           ## (N, 32, 64, 64)

        # x = self.relu(x)          #(N, 64,  64, 64)  
        # x = self.head1(x)
        x = unsample1(x)         #(N, 64,  128, 128) 
        
        b = (self.maxpool(b)).view(-1, 16, 128, 128)
        x = torch.cat((x,b),dim=1)       
        x = self.de_block2(x)

        # x = self.relu(x)          #(N, 32, 128, 128) 
        
        # x = self.conv8(x)       #(N, 32, 128, 128) 
        x = self.conv9(x)       #(N, 3, 128, 128)
        x = self.sigm(x)        #(N, 3, 128, 128)
        return x  

class MultiResUNet1_base(nn.Module):
    def __init__(self):
        super(MultiResUNet1_base, self).__init__()
        self.conv0 = nn.Conv3d(2, 16, (1, 1, 1), stride=1, padding=(0, 0, 0))
        # self.en_block1 = Multiresblock_1(16)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        # self.conv1 = nn.Conv3d(16, 32, (4, 4, 4), stride=2, padding=(1, 1, 1))
        self.conv1 = nn.Conv3d(48, 16, (1, 1, 1), stride=1, padding=(0, 0, 0))
        # self.en_block2 = Multiresblock_1(48)
        self.conv2 = nn.Conv3d(144, 64, (1, 1, 1), stride=1, padding=(0, 0, 0))
        # self.conv2 = nn.Conv3d(32, 64, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.en_block3 = Multiresblock_1(64)
        self.head = DAHead_backbone192(in_channels=192)
        # self.head1 = DAHead_backbone64(in_channels=64,nclass=64)
        # self.de_block1 = Multiresblock2D_1(32,192,32)
        # self.de_block2 = Multiresblock2D1_1(16,64)
        self.relu = nn.PReLU()
        self.conv3 = nn.Conv3d(64, 32, (1, 1, 1), stride=1, padding=(0, 0, 0))
        # self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1))
        self.maxpoolc = nn.MaxPool3d(kernel_size=(2, 1, 1))

        
        # self.conv8 = nn.Conv2d(64,16,( 3, 3), stride=1, padding=( 1, 1))
        self.conv9 = nn.Conv2d(32,3,( 1, 1), stride=1, padding=( 0, 0))
        # self.convk = nn.Conv2d(16,32,( 3, 3), stride=1, padding=( 1, 1))
        self.sigm = nn.Sigmoid()
        self.conv1_1 = nn.Conv3d(16, 32, (3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv1_2 = nn.Conv3d(32, 48, (3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv2_1 = nn.Conv3d(48, 96, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv2_2 = nn.Conv3d(96, 144, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv3_1 = nn.Conv3d(64, 128, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv3_2 = nn.Conv3d(128, 192, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv4_1 = nn.Conv2d(192, 64, (3, 3), stride=1, padding=( 1, 1))
        self.conv4_2 = nn.Conv2d(64, 32, (3, 3), stride=1, padding=( 1, 1))
        self.conv5_1 = nn.Conv2d(64, 32, (3, 3), stride=1, padding=( 1, 1))
        self.conv5_2 = nn.Conv2d(32, 16, (3, 3), stride=1, padding=( 1, 1))

    def forward(self, x):
        # batch_size, in_chirps, n_channels, w, h = x.shape
        
        x = self.conv0(x)
        x = self.conv1_1(x)       #(N, 48, 4, 128, 128)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.relu(x)
        # x = self.conv1(x)           #(N, 48, 4, 128, 128)
        b = x
        x = self.pool1(x)           # (N, 48, 2, 64, 64)
        x = self.conv2_1(x) 
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.relu(x)       # (N, 144, 2, 64, 64)
        x= self.conv2(x)            # (N, 64, 2, 64, 64)
        c = x
        x = self.pool1(x)           # (N, 64, 1, 32, 32)
        # x = self.en_block3(x)       # (N, 192, 1, 32, 32)
        x = self.conv3_1(x)       
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        x = x.view(-1, 192, 32, 32)
        # x = self.head(x)        # (N, 192, 32, 32)


        x = unsample1(x)                 # (N, 192, 64, 64)

        
        # x = self.de_block1(x)           ## (N, 32, 64, 64)
        x = self.conv4_1(x)       
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.relu(x)
        c = (self.maxpoolc(self.conv3(c))).view(-1, 32, 64, 64)
        x = torch.cat((x,c),dim=1)
        # x = self.relu(x)          #(N, 64,  64, 64)  
        # x = self.head1(x)
        x = unsample1(x)         #(N, 64,  128, 128) 
        
        # x = self.de_block2(x)
        x = self.conv5_1(x)       
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.relu(x)
        b = (self.maxpool(self.conv1(b))).view(-1, 16, 128, 128)
        x = torch.cat((x,b),dim=1)
        # x = self.relu(x)          #(N, 32, 128, 128) 
        
        # x = self.conv8(x)       #(N, 32, 128, 128) 
        x = self.conv9(x)       #(N, 3, 128, 128)
        x = self.sigm(x)        #(N, 3, 128, 128)
        return x  

class MultiResUNet_2D_1(nn.Module):
    def __init__(self, alpha=1):
        super(MultiResUNet_2D_1, self).__init__()
        # self.conv0 = nn.Conv3d(2, 16, (1, 1, 1), stride=1, padding=(0, 0, 0))
        
        self.alpha = alpha
        self.en_block1 = Multiresblock_3D_2D(16,2)    
        self.en_block11 = Multiresblock_3D_2D(32,32)  
        self.en_1 = nn.Conv2d(64, 32, 1,1,0)      
        self.in_filters1 = round(32*self.alpha*0.167)+round(32*self.alpha*0.333)+round(32*self.alpha* 0.5)
        self.pool1 = nn.Conv2d(32,self.in_filters1,3,2,1)
        self.respath1 = Respath(self.in_filters1,32,respath_length=4)

        self.en_block2 = Multiresblock_conv(self.in_filters1,32*2)
        self.in_filters2 = round(32*2*self.alpha*0.167)+round(32*2*self.alpha*0.333)+round(32*2*self.alpha* 0.5)
        self.pool2 =  nn.Conv2d(64,64,3,2,1) 
        self.respath2 = Respath(self.in_filters2,32*2,respath_length=3)

        self.en_block3 =  Multiresblock_conv(self.in_filters2,32*4)
        self.in_filters3 = round(32*4*self.alpha*0.167)+round(32*4*self.alpha*0.333)+round(32*4*self.alpha* 0.5)
        self.pool3 =  nn.Conv2d(128,128,3,2,1)
        self.respath3 = Respath(self.in_filters3,32*4,respath_length=2)
        
        # self.en_block4 = Multiresblock_conv(self.in_filters3,32*8)
        # self.in_filters4 = round(32*8*self.alpha*0.167)+round(32*8*self.alpha*0.333)+round(32*8*self.alpha* 0.5)
        self.head = DAHead_backbone256(in_channels=128,nclass=256)
        # self.respath4 = Respath(self.in_filters4,32*8,respath_length=1)
       
        # self.in_filters6 = round(32*8*self.alpha*0.167)+round(32*8*self.alpha*0.333)+round(32*8*self.alpha* 0.5)

         
        # self.concat_filters2 = 256
        # self.de_block1 = Multiresblock_conv(self.concat_filters2,32*4)
        # self.in_filters7 = round(32*4*self.alpha*0.167)+round(32*4*self.alpha*0.333)+round(32*4*self.alpha* 0.5)

        # self.upsample8 = torch.nn.ConvTranspose2d(self.in_filters7,32*2,kernel_size=(2,2),stride=(2,2))
        self.concat_filters3 = 192
        self.de_block2 = Multiresblock_conv(self.concat_filters3,32*2)
        self.in_filters8 = round(32*2*self.alpha*0.167)+round(32*2*self.alpha*0.333)+round(32*2*self.alpha* 0.5)

        # self.upsample9 = torch.nn.ConvTranspose2d(self.in_filters8,32,kernel_size=(2,2),stride=(2,2))
        self.concat_filters4 = 96
        self.de_block3 = Multiresblock_conv(self.concat_filters4,32)
        self.in_filters9 = round(32*self.alpha*0.167)+round(32*self.alpha*0.333)+round(32*self.alpha* 0.5)

        self.conv_final = Conv2d_batchnorm(self.in_filters9, 3, kernel_size = (1,1), padding=0,activation='None')

        self.relu = nn.PReLU()
        self.sigm = nn.Sigmoid()
    def forward(self, x):
        # batch_size, in_chirps, n_channels, w, h = x.shape
        
        
        x = self.en_block1(x)       #(N, 32, 128, 128)
        x = self.en_block11(x).view(-1, 64, 128, 128)
        x = self.en_1(x)            #(N, 32, 128, 128)
        b = x
        
        x = self.pool1(x)           # (N, 32,  64, 64)
        x = self.relu(x)
        x = self.en_block2(x)       # (N, 64,  64, 64)
        c = x
        
        x = self.pool2(x)           # (N, 64,  32, 32) 
        x = self.relu(x)
        x = self.en_block3(x)       # (N, 128,  32, 32)
        # d = x


        
        x = self.head(x)        # (N, 128, 32, 32)
        x = unsample1(x)                 # (N, 128, 64, 64)

        # d = self.respath3(d)             #(N, 128, 32, 32)
        # x = torch.cat((x,d),dim=1)       #   (N, 256, 32, 32)    
        # x = self.de_block1(x)           ## (N, 128, 32, 32)

        # x = unsample1(x)                 # (N, 128, 64, 64)

        c = self.respath2(c)            #(N, 64, 64, 64)        
        x = torch.cat((x,c),dim=1)        #(N, 192, 64, 64)
        x = self.de_block2(x)           ## (N, 64, 64, 64)

        # x = self.relu(x)          #(N, 64,  64, 64)  
        
        x = unsample1(x)         #(N, 64,  128, 128) 
        
        b = self.respath1(b)        #(N, 32,  128, 128) 
        
        x = torch.cat((x,b),dim=1)      #(N, 96,  128, 128)
        x = self.de_block3(x)       #(N, 32,  128, 128)
        
        # x = self.relu(x)          #(N, 32, 128, 128) 
        
        # x = self.conv8(x)       #(N, 32, 128, 128) 
        x = self.conv_final(x)       #(N, 3, 128, 128)
        x = self.sigm(x)        #(N, 3, 128, 128)
        return x  

if __name__ == '__main__':
    backbone       = MultiResUNet1_1().cuda()


    input = torch.randn(1,2,4, 128, 128).cuda()
    out =  backbone(input)

    print(out.shape)
    
    
    
