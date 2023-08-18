import torch.nn as nn
import torch
# from rodnet.models.model_UHRNet import UHRnet
# from rodnet.models.DANet import DAHead
from rodnet.models.u2net import U2NET3  #U2NET,U2NETP,
# from rodnet.models.modules.mnet import MNet,BackboneNet,ROD_RA,BackboneNet_1,ROD_RA_1,BackboneNet_Head,BackboneNet_Head_V1,BackboneNet_Head_V2,BackboneNet_Head_V3
# from rodnet.models.modules.mnet import BackboneNet_Dilation_V1,BackboneNet_Dilation,BackboneNet_Dilation_V2,BackboneNet_Head_V1_dilation,BackboneNet_Dilation_V5
from rodnet.models.modules.mnet import MNet,MultiResUNet1,MultiResUNet1_base,BackboneNet_MAX_V3_1
class RODNetUNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=3+1, mnet_cfg=(4,3), backbone = "UHRNet_W18_Small"):
        super(RODNetUNet, self).__init__()
        self.backbone =backbone
        self.num_classes = num_classes
        # self.conv_op = nn.Conv3d
        if mnet_cfg is not None:
            in_chirps_mnet, out_channels_mnet = mnet_cfg
            assert in_channels == in_chirps_mnet
            # self.mnet = MNet(in_chirps_mnet, out_channels_mnet)
            self.mnet = MultiResUNet1_base()
            self.with_mnet = True
            
            self.cdc = MNet()                            #U2NET3(32,3)     UHRnet()  DAHead(in_channels=32,nclass=3) 
        else:
            self.with_mnet = False
            self.cdc = U2NET3(num_classes=self.num_classes, backbone = self.backbone)

    def forward(self, x):
        if self.with_mnet:
            # x = x.permute(0, 2, 1, 3, 4)
            x = self.mnet(x)  #, out1, out2, out1, out2
        x = self.cdc(x)
        return x

if __name__ == '__main__':
    
    from rodnet.models.M_UHRNet import RODNetUNet 
    torch.cuda.set_device(torch.device('cuda:1'))
    batch_size = 8
    in_channels = 2
    # win_size = 4
    in_chirps = 4
    w = 128
    h = 128
    out_channels = 8
    model = RODNetUNet(num_classes=3, mnet_cfg=(4,6), backbone = "UHRNet_W48").cuda()
    for iter in range(10):
        input = torch.randn(batch_size, in_chirps, in_channels, w, h).cuda()
        output = model(input)
        
        print("forward done")
        output_gt = torch.randn(batch_size, 3, w, h).cuda()
        criterion = nn.BCELoss()
        loss = criterion(output, output_gt)
        print("loss:",loss)
        loss.backward()
