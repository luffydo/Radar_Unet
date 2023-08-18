class BackboneNet(nn.Module):
    def __init__(self):
        super(BackboneNet, self).__init__()
        self.conv1 = nn.Conv3d(2, 16, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool1 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv2 = nn.Conv3d(16, 32, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv3 = nn.Conv3d(32, 64, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool3 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv4 = nn.Conv3d(64, 128, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool4 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv5 = nn.Conv3d(2,3,(1,1,1),stride=(1,1,1),padding=(0,0,0))
        self.relu = nn.LeakyReLU()
        self.deconv1 = nn.ConvTranspose3d(64, 32, (2, 2, 2), stride=(2, 2, 2))
        self.deconv2 = nn.ConvTranspose3d(32, 16, (2, 2, 2), stride=(2, 2, 2))
        self.deconv3 = nn.ConvTranspose3d(16, 16, (2, 2, 2), stride=(2, 2, 2))
        self.upsample = nn.Upsample(size=(4, 128, 128), mode='nearest')
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1))
    def forward(self, x):
        batch_size, in_chirps, n_channels, w, h = x.shape
        b = x
        x = self.conv1(x)           # (N, 16, win, r, a)
        # x = self.pool1(x)           # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.conv2(x)           # (N, 32, win/2, r/2, a/2)
        # x = self.pool2(x)           # (N, 32, win/4, r/4, a/4)
        x = self.relu(x)
        # x = self.conv3(x)           # (N, 64, win/4, r/4, a/4)
        # # x = self.pool3(x)           # (N, 64, win/8, r/8, a/8)
        # x = self.relu(x)
        # x0_np = x.cpu().detach().numpy()[0, 0, 0, :, :]
        # x1_np = x.cpu().detach().numpy()[0, 1, 0, :, :]
        # x2_np = x.cpu().detach().numpy()[0, 2, 0, :, :]
                # x = self.deconv1(x)         # (N, 32, win/4, r/4, a/4)
        # x = self.relu(x)
        x = self.deconv2(x)         # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.deconv3(x)         # (N, n_class, win, r, a)
        x = self.relu(x)
        x = x + self.conv5(b)
        # x = self.upsample(x)
        # x0_np = x.cpu().detach().numpy()[0, 0, 0, :, :]
        # x1_np = x.cpu().detach().numpy()[0, 1, 0, :, :]
        # x2_np = x.cpu().detach().numpy()[0, 2, 0, :, :]
        x = self.sigmoid(x)
        x = self.maxpool(x)
        x = x.view(batch_size, 3, 128, 128)
        return x
class BackboneNet_1(nn.Module):
    def __init__(self):
        super(BackboneNet_1, self).__init__()
        self.conv1 = nn.Conv3d(2, 16, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool1 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv2 = nn.Conv3d(16, 32, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv3 = nn.Conv3d(32, 64, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool3 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv4 = nn.Conv3d(64, 128, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool4 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv5 = nn.Conv3d(2,16,(1,1,1),stride=(1,1,1),padding=(0,0,0))
        self.relu = nn.LeakyReLU()
        self.deconv1 = nn.ConvTranspose3d(64, 32, (2, 2, 2), stride=(2, 2, 2))
        self.deconv2 = nn.ConvTranspose3d(32, 32, (2, 2, 2), stride=(2, 2, 2))
        self.deconv3 = nn.ConvTranspose3d(32, 16, (2, 2, 2), stride=(2, 2, 2))
        self.upsample = nn.Upsample(size=(4, 128, 128), mode='nearest')
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1))
        self.avg = nn.AdaptiveAvgPool3d((1,128,128))
    def forward(self, x):
        batch_size, in_chirps, n_channels, w, h = x.shape
        b = x
        x = self.conv1(x)           # (N, 16, win, r, a)
        # x = self.pool1(x)           # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.conv2(x)           # (N, 32, win/2, r/2, a/2)
        # x = self.pool2(x)           # (N, 32, win/4, r/4, a/4)
        x = self.relu(x)
        # x = self.conv3(x)           # (N, 64, win/4, r/4, a/4)
        # # x = self.pool3(x)           # (N, 64, win/8, r/8, a/8)
        # x = self.relu(x)
        # x0_np = x.cpu().detach().numpy()[0, 0, 0, :, :]
        # x1_np = x.cpu().detach().numpy()[0, 1, 0, :, :]
        # x2_np = x.cpu().detach().numpy()[0, 2, 0, :, :]
                # x = self.deconv1(x)         # (N, 32, win/4, r/4, a/4)
        # x = self.relu(x)
        x = self.deconv2(x)         # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.deconv3(x)         # (N, n_class, win, r, a)
        x = self.relu(x)
        x = x + self.conv5(b)
        # x = self.upsample(x)
        # x0_np = x.cpu().detach().numpy()[0, 0, 0, :, :]
        # x1_np = x.cpu().detach().numpy()[0, 1, 0, :, :]
        # x2_np = x.cpu().detach().numpy()[0, 2, 0, :, :]
        # x = self.sigmoid(x)
        b = x
        x = self.avg(x)
        b = self.maxpool(b)
        x = torch.cat((x,b),dim=1)
        # x = self.sigmoid(x)
        x = x.view(batch_size, 32, 128, 128)
        return x
class ROD_RA(nn.Module):
    
    def __init__(self):
        super(ROD_RA, self).__init__()
        self.conv1a = nn.Conv3d(in_channels=2, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv1b = nn.Conv3d(in_channels=64, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv2a = nn.Conv3d(in_channels=64, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv2b = nn.Conv3d(in_channels=128, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv3a = nn.Conv3d(in_channels=128, out_channels=256,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv3b = nn.Conv3d(in_channels=256, out_channels=256,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.convt1 = nn.ConvTranspose3d(in_channels=256, out_channels=128,
                                         kernel_size=(1, 6, 6), stride=(2, 2, 2), padding=(0, 2, 2))
        self.convt2 = nn.ConvTranspose3d(in_channels=128, out_channels=64,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt3 = nn.ConvTranspose3d(in_channels=64, out_channels=32,
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        self.convpool = nn.Conv3d(in_channels=32, out_channels=32,kernel_size=(2, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        # self.conv4a = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        # self.conv4b = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        # self.conv5a = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        # self.conv5b = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        self.bn1a = nn.BatchNorm3d(num_features=64)
        self.bn1b = nn.BatchNorm3d(num_features=64)
        self.bn2a = nn.BatchNorm3d(num_features=128)
        self.bn2b = nn.BatchNorm3d(num_features=128)
        self.bn3a = nn.BatchNorm3d(num_features=256)
        self.bn3b = nn.BatchNorm3d(num_features=256)
        # self.bn4a = nn.BatchNorm3d(num_features=64)
        # self.bn4b = nn.BatchNorm3d(num_features=64)
        # self.bn5a = nn.BatchNorm3d(num_features=64)
        # self.bn5b = nn.BatchNorm3d(num_features=64)
        self.relu = nn.ReLU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(size=(4, 128, 128), mode='nearest')
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 1, 1))
    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))  # (B, 2, W, 128, 128) -> (B, 64, W, 128, 128) Note: W~2W in this case
        x = self.relu(self.bn1b(self.conv1b(x)))  # (B, 64, W, 128, 128) -> (B, 64, W/2, 64, 64)
        x = self.relu(self.bn2a(self.conv2a(x)))  # (B, 64, W/2, 64, 64) -> (B, 128, W/2, 64, 64)
        x = self.relu(self.bn2b(self.conv2b(x)))  # (B, 128, W/2, 64, 64) -> (B, 128, W/4, 32, 32)
        x = self.relu(self.bn3a(self.conv3a(x)))  # (B, 128, W/4, 32, 32) -> (B, 256, W/4, 32, 32)
        x = self.relu(self.bn3b(self.conv3b(x)))  # (B, 256, W/4, 32, 32) -> (B, 256, W/8, 16, 16)
        # x = self.relu(self.bn4a(self.conv4a(x)))
        # x = self.relu(self.bn4b(self.conv4b(x)))
        # x = self.relu(self.bn5a(self.conv5a(x)))
        # x = self.relu(self.bn5b(self.conv5b(x)))
        x = self.prelu(self.convt1(x))  # (B, 256, W/8, 16, 16) -> (B, 128, W/4, 32, 32)
        x = self.prelu(self.convt2(x))  # (B, 128, W/4, 32, 32) -> (B, 64, W/2, 64, 64)
        x = self.prelu(self.convt3(x))  # (B, 64, W/2, 64, 64) -> (B, 32, W/2, 128, 128)
        x = self.convpool(x)
        x = x.view(-1, 32, 128, 128)
        return x
    
class ROD_RA_1(nn.Module):
    
    def __init__(self):
        super(ROD_RA_1, self).__init__()
        self.conv1a = nn.Conv3d(in_channels=2, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv1b = nn.Conv3d(in_channels=64, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv2a = nn.Conv3d(in_channels=64, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv2b = nn.Conv3d(in_channels=128, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv3a = nn.Conv3d(in_channels=128, out_channels=256,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv3b = nn.Conv3d(in_channels=256, out_channels=256,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.convt1 = nn.ConvTranspose3d(in_channels=256, out_channels=128,
                                         kernel_size=(1, 6, 6), stride=(2, 2, 2), padding=(0, 2, 2))
        self.convt2 = nn.ConvTranspose3d(in_channels=128, out_channels=64,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt3 = nn.ConvTranspose3d(in_channels=64, out_channels=32,
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        self.convpool = nn.Conv3d(in_channels=32, out_channels=32,kernel_size=(2, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        # self.conv4a = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        # self.conv4b = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        # self.conv5a = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        # self.conv5b = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        self.bn1a = nn.BatchNorm3d(num_features=64)
        self.bn1b = nn.BatchNorm3d(num_features=64)
        self.bn2a = nn.BatchNorm3d(num_features=128)
        self.bn2b = nn.BatchNorm3d(num_features=128)
        self.bn3a = nn.BatchNorm3d(num_features=256)
        self.bn3b = nn.BatchNorm3d(num_features=256)
        # self.bn4a = nn.BatchNorm3d(num_features=64)
        # self.bn4b = nn.BatchNorm3d(num_features=64)
        # self.bn5a = nn.BatchNorm3d(num_features=64)
        # self.bn5b = nn.BatchNorm3d(num_features=64)
        self.relu = nn.ReLU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(size=(4, 128, 128), mode='nearest')
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 1, 1))
        self.avg = nn.AdaptiveAvgPool3d((1,128,128))
    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))  # (B, 2, W, 128, 128) -> (B, 64, W, 128, 128) Note: W~2W in this case
        x = self.relu(self.bn1b(self.conv1b(x)))  # (B, 64, W, 128, 128) -> (B, 64, W/2, 64, 64)
        x = self.relu(self.bn2a(self.conv2a(x)))  # (B, 64, W/2, 64, 64) -> (B, 128, W/2, 64, 64)
        x = self.relu(self.bn2b(self.conv2b(x)))  # (B, 128, W/2, 64, 64) -> (B, 128, W/4, 32, 32)
        x = self.relu(self.bn3a(self.conv3a(x)))  # (B, 128, W/4, 32, 32) -> (B, 256, W/4, 32, 32)
        x = self.relu(self.bn3b(self.conv3b(x)))  # (B, 256, W/4, 32, 32) -> (B, 256, W/8, 16, 16)
        # x = self.relu(self.bn4a(self.conv4a(x)))
        # x = self.relu(self.bn4b(self.conv4b(x)))
        # x = self.relu(self.bn5a(self.conv5a(x)))
        # x = self.relu(self.bn5b(self.conv5b(x)))
        x = self.prelu(self.convt1(x))  # (B, 256, W/8, 16, 16) -> (B, 128, W/4, 32, 32)
        x = self.prelu(self.convt2(x))  # (B, 128, W/4, 32, 32) -> (B, 64, W/2, 64, 64)
        x = self.prelu(self.convt3(x))  # (B, 64, W/2, 64, 64) -> (B, 32, 2, 128, 128)
        b = x
        x = self.avg(x)

        b = self.maxpool(b)

        x = torch.cat((x,b),dim=1)

        x = x.view(-1, 64, 128, 128)
        return x
    
class BackboneNet_Head(nn.Module):
    def __init__(self):
        super(BackboneNet_Head, self).__init__()
        self.conv1 = nn.Conv3d(2, 16, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool1 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv2 = nn.Conv3d(16, 32, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv3 = nn.Conv3d(32, 64, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool3 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv4 = nn.Conv3d(64, 128, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool4 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv5 = nn.Conv3d(2,16,(1,1,1),stride=(1,1,1),padding=(0,0,0))
        self.relu = nn.LeakyReLU()
        self.deconv1 = nn.ConvTranspose3d(64, 32, (2, 2, 2), stride=(2, 2, 2))
        self.deconv2 = nn.ConvTranspose3d(32, 32, (2, 2, 2), stride=(2, 2, 2))
        self.deconv3 = nn.ConvTranspose3d(32, 16, (2, 2, 2), stride=(2, 2, 2))
        self.upsample = nn.Upsample(size=(4, 128, 128), mode='nearest')
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1))
        self.avg = nn.AdaptiveAvgPool3d((1,128,128))
        self.head = DAHead_backbone(in_channels=32,nclass=32)
        # self.conv6 = nn.Conv3d(16,32,(4,1,1),stride=(1,1,1),padding=(0,0,0))
    def forward(self, x):
        batch_size, in_chirps, n_channels, w, h = x.shape
        b = x
        x = self.conv1(x)           # (N, 16, win, r, a)
        # x = self.pool1(x)           # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.conv2(x)           # (N, 32, win/2, r/2, a/2)
        # x = self.pool2(x)           # (N, 32, win/4, r/4, a/4)
        x = self.relu(x)
        x = x.view(batch_size, 32, 32, 32)
        x = self.head(x)
        x = x.view(batch_size, 32,1, 32, 32)
        # x = self.conv3(x)           # (N, 64, win/4, r/4, a/4)
        # # x = self.pool3(x)           # (N, 64, win/8, r/8, a/8)
        # x = self.relu(x)
        # x0_np = x.cpu().detach().numpy()[0, 0, 0, :, :]
        # x1_np = x.cpu().detach().numpy()[0, 1, 0, :, :]
        # x2_np = x.cpu().detach().numpy()[0, 2, 0, :, :]
                # x = self.deconv1(x)         # (N, 32, win/4, r/4, a/4)
        # x = self.relu(x)
        x = self.deconv2(x)         # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.deconv3(x)         # (N, n_class, win, r, a)
        x = self.relu(x)
        x = x + self.conv5(b)
        # x = self.upsample(x)
        # x0_np = x.cpu().detach().numpy()[0, 0, 0, :, :]
        # x1_np = x.cpu().detach().numpy()[0, 1, 0, :, :]
        # x2_np = x.cpu().detach().numpy()[0, 2, 0, :, :]
        # x = self.sigmoid(x)
        b = x
        # c = x
        x = self.avg(x)
        b = self.maxpool(b)
        x = torch.cat((x,b),dim=1)
        # x = x  + self.conv6(c)
        # x = self.sigmoid(x)
        x = x.view(batch_size, 32, 128, 128)
        return x    

class BackboneNet_Head_V1(nn.Module):
    def __init__(self):
        super(BackboneNet_Head_V1, self).__init__()
        self.conv1 = nn.Conv3d(2, 16, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool1 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv2 = nn.Conv3d(16, 32, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        # self.conv3 = nn.Conv3d(32, 64, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool3 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        # self.conv4 = nn.Conv3d(64, 128, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool4 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv5 = nn.Conv3d(2,16,(1,1,1),stride=(1,1,1),padding=(0,0,0))
        self.relu = nn.PReLU()
        # self.deconv1 = nn.ConvTranspose3d(64, 32, (2, 2, 2), stride=(2, 2, 2))
        self.deconv2 = nn.ConvTranspose3d(32, 32, (2, 2, 2), stride=(2, 2, 2))
        self.deconv3 = nn.ConvTranspose3d(32, 16, (2, 2, 2), stride=(2, 2, 2))

        # self.sigmoid = nn.Sigmoid()
        # self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1))
        self.avg = nn.AdaptiveAvgPool3d((1,128,128))
        self.head = DAHead_backbone(in_channels=32,nclass=32)
        self.conv6 = nn.Conv3d(16,32,(4,1,1),stride=(1,1,1),padding=(0,0,0))
    def forward(self, x):
        batch_size, in_chirps, n_channels, w, h = x.shape
        b = x
        x = self.conv1(x)           # (N, 16, win, r, a)
        # x = self.pool1(x)           # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.conv2(x)           # (N, 32, win/2, r/2, a/2)
        # x = self.pool2(x)           # (N, 32, win/4, r/4, a/4)
        x = self.relu(x)
        x = x.view(batch_size, 32, 32, 32)
        x = self.head(x)
        x = x.view(batch_size, 32,1, 32, 32)
        # x = self.conv3(x)           # (N, 64, win/4, r/4, a/4)
        # # x = self.pool3(x)           # (N, 64, win/8, r/8, a/8)
        # x = self.relu(x)
        # x0_np = x.cpu().detach().numpy()[0, 0, 0, :, :]
        # x1_np = x.cpu().detach().numpy()[0, 1, 0, :, :]
        # x2_np = x.cpu().detach().numpy()[0, 2, 0, :, :]
                # x = self.deconv1(x)         # (N, 32, win/4, r/4, a/4)
        # x = self.relu(x)
        x = self.deconv2(x)         # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.deconv3(x)         # (N, n_class, win, r, a)
        x = self.relu(x)
        x = x + self.conv5(b)
        # x = self.upsample(x)
        # x0_np = x.cpu().detach().numpy()[0, 0, 0, :, :]
        # x1_np = x.cpu().detach().numpy()[0, 1, 0, :, :]
        # x2_np = x.cpu().detach().numpy()[0, 2, 0, :, :]
        # x = self.sigmoid(x)
        b = x
        c = x
        x = self.avg(x)
        b = self.maxpool(b)
        x = torch.cat((x,b),dim=1)
        x = x  + self.conv6(c)
        # x = self.sigmoid(x)
        x = x.view(batch_size, 32, 128, 128)
        return x  


class BackboneNet_Head_V2(nn.Module):

    def __init__(self):
        super(BackboneNet_Head_V2, self).__init__()
        self.conv1 = nn.Conv3d(2, 16, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool1 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv2 = nn.Conv3d(16, 32, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv3 = nn.Conv3d(32, 64, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool3 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv4 = nn.Conv3d(64, 128, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool4 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv5 = nn.Conv3d(2,32,(1,1,1),stride=(1,1,1),padding=(0,0,0))
        self.relu = nn.LeakyReLU()
        self.deconv1 = nn.ConvTranspose3d(64, 32, (2, 2, 2), stride=(2, 2, 2))
        self.deconv2 = nn.ConvTranspose3d(32, 32, (2, 2, 2), stride=(2, 2, 2))
        self.deconv3 = nn.ConvTranspose3d(32, 32, (2, 2, 2), stride=(2, 2, 2))
        self.upsample = nn.Upsample(size=(4, 128, 128), mode='nearest')
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1))
        self.avg = nn.AdaptiveAvgPool3d((1,128,128))
        self.head = DAHead_backbone(in_channels=32,nclass=32)
        self.conv6 = nn.Conv3d(32,32,(4,1,1),stride=(1,1,1),padding=(0,0,0))
    def forward(self, x):
        batch_size, in_chirps, n_channels, w, h = x.shape
        b = x
        x = self.conv1(x)           # (N, 16, win, r, a)
        # x = self.pool1(x)           # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.conv2(x)           # (N, 32, win/2, r/2, a/2)
        # x = self.pool2(x)           # (N, 32, win/4, r/4, a/4)
        x = self.relu(x)
        x = x.view(batch_size, 32, 32, 32)
        x = self.head(x)
        x = x.view(batch_size, 32,1, 32, 32)
        # x = self.conv3(x)           # (N, 64, win/4, r/4, a/4)
        # # x = self.pool3(x)           # (N, 64, win/8, r/8, a/8)
        # x = self.relu(x)
        # x0_np = x.cpu().detach().numpy()[0, 0, 0, :, :]
        # x1_np = x.cpu().detach().numpy()[0, 1, 0, :, :]
        # x2_np = x.cpu().detach().numpy()[0, 2, 0, :, :]
                # x = self.deconv1(x)         # (N, 32, win/4, r/4, a/4)
        # x = self.relu(x)
        x = self.deconv2(x)         # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.deconv3(x)         # (N, n_class, win, r, a)
        x = self.relu(x)
        x = x + self.conv5(b)
        # x = self.upsample(x)
        # x0_np = x.cpu().detach().numpy()[0, 0, 0, :, :]
        # x1_np = x.cpu().detach().numpy()[0, 1, 0, :, :]
        # x2_np = x.cpu().detach().numpy()[0, 2, 0, :, :]
        # x = self.sigmoid(x)
        
        
        
        x = self.avg(x)
        
        
        
        
        # x = self.sigmoid(x)
        x = x.view(batch_size, 32, 128, 128)
        return x

class BackboneNet_Head_V3(nn.Module):
    def __init__(self):
        super(BackboneNet_Head_V3, self).__init__()
        self.conv1 = nn.Conv3d(2, 16, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool1 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv2 = nn.Conv3d(16, 32, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv3 = nn.Conv3d(32, 64, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool3 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv4 = nn.Conv3d(64, 128, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool4 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv5 = nn.Conv3d(2,16,(1,1,1),stride=(1,1,1),padding=(0,0,0))
        self.relu = nn.LeakyReLU()
        self.deconv1 = nn.ConvTranspose3d(64, 32, (2, 2, 2), stride=(2, 2, 2))
        self.deconv2 = nn.ConvTranspose3d(32, 32, (2, 2, 2), stride=(2, 2, 2))
        self.deconv3 = nn.ConvTranspose3d(32, 16, (2, 2, 2), stride=(2, 2, 2))
        self.upsample = nn.Upsample(size=(4, 128, 128), mode='nearest')
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1))
        self.avg = nn.AdaptiveAvgPool3d((1,128,128))
        self.head = DAHead_backbone(in_channels=32,nclass=32)
        self.conv6 = nn.Conv3d(16,32,(4,1,1),stride=(1,1,1),padding=(0,0,0))
        
        
        self.t_conv3d = nn.Conv3d(in_channels=2, out_channels=12, kernel_size=(3, 1, 1), stride=(2, 1, 1),
                    padding=(2, 0, 0))
        self.c_conv3d = nn.Conv3d(in_channels=12, out_channels=6, kernel_size=(2, 1, 1), stride=(1, 1, 1),
                    padding=(0, 0, 0))
        self.a_conv3d = nn.Conv3d(in_channels=2, out_channels=6, kernel_size=(1, 1, 1), stride=(2, 1, 1),
                padding=(0, 0, 0))
        # t_conv_out = math.floor((in_chirps + 2 * 1 - (3 - 1) - 1) / 2 + 1)
        # self.t_maxpool = nn.MaxPool3d(kernel_size=(t_conv_out, 1, 1))
        self.conv2d = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=( 1, 1), stride= 1)
        self.t_maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1))
    def forward(self, x):
        batch_size, in_chirps, n_channels, w, h = x.shape
        b = x
        
        x_win = self.t_conv3d(x)                            #4,3,2,128,128
        x_win = self.c_conv3d(x_win)
        x_win = torch.cat((x_win,self.a_conv3d(x)),dim=2) 
        x_win = self.t_maxpool(x_win)                       #4,3,1,128,128
        x_win = x_win.view(-1, 6, 128, 128)   #4,3,128,128     
        x_win = self.conv2d(x_win)
        
        x = self.conv1(x)           # (N, 16, win, r, a)
        # x = self.pool1(x)           # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.conv2(x)           # (N, 32, win/2, r/2, a/2)
        # x = self.pool2(x)           # (N, 32, win/4, r/4, a/4)
        x = self.relu(x)
        x = x.view(batch_size, 32, 32, 32)
        x = self.head(x)
        x = x.view(batch_size, 32,1, 32, 32)
        # x = self.conv3(x)           # (N, 64, win/4, r/4, a/4)
        # # x = self.pool3(x)           # (N, 64, win/8, r/8, a/8)
        # x = self.relu(x)
        # x0_np = x.cpu().detach().numpy()[0, 0, 0, :, :]
        # x1_np = x.cpu().detach().numpy()[0, 1, 0, :, :]
        # x2_np = x.cpu().detach().numpy()[0, 2, 0, :, :]
                # x = self.deconv1(x)         # (N, 32, win/4, r/4, a/4)
        # x = self.relu(x)
        x = self.deconv2(x)         # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.deconv3(x)         # (N, n_class, win, r, a)
        x = self.relu(x)
        x = x + self.conv5(b)
        # x = self.upsample(x)
        # x0_np = x.cpu().detach().numpy()[0, 0, 0, :, :]
        # x1_np = x.cpu().detach().numpy()[0, 1, 0, :, :]
        # x2_np = x.cpu().detach().numpy()[0, 2, 0, :, :]
        # x = self.sigmoid(x)
        b = x
        c = x
        x = self.avg(x)
        b = self.maxpool(b)
        x = torch.cat((x,b),dim=1)
        x = x  + self.conv6(c)
        # x = self.sigmoid(x)
        x = x.view(batch_size, 32, 128, 128)
        x = torch.cat((x,x_win),dim=1)
        return x  


class BackboneNet_Dilation(nn.Module):
    def __init__(self):
        super(BackboneNet_Dilation, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=0,dilation=2)
        self.relu = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=0,dilation=3)
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,dilation=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,stride=2,output_padding=1,dilation=3)
        self.conv4 = nn.Conv2d(in_channels=8,out_channels=32,kernel_size=1,stride=1)

        self.head = DAHead_backbone64(in_channels=64,nclass=64)

    def forward(self, x):

        x = x.reshape(-1, 8, 128, 128)
        b = x
        x = self.conv1(x)           # (N, 16, win, r, a)
        # x = self.pool1(x)           # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.conv2(x)           # (N, 32, win/2, r/2, a/2)
        # x = self.pool2(x)           # (N, 32, win/4, r/4, a/4)
        x = self.relu(x)
        x = self.conv3(x) 
        x = self.relu(x)
        x = self.head(x)
        
        x = self.deconv1(x)         # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.deconv2(x)         # (N, n_class, win, r, a)
        x = self.relu(x)
        x = x + self.conv4(b)
        # x = self.upsample(x)
        # x0_np = x.cpu().detach().numpy()[0, 0, 0, :, :]
        # x1_np = x.cpu().detach().numpy()[0, 1, 0, :, :]
        # x2_np = x.cpu().detach().numpy()[0, 2, 0, :, :]
        # x = self.sigmoid(x)


        return x     #b,32,128,128

class BackboneNet_Dilation_V1(nn.Module):
    def __init__(self):
        super(BackboneNet_Dilation_V1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=0,dilation=2)
        self.relu = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=0,dilation=3)
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,dilation=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,stride=2,output_padding=1,dilation=3)
        self.conv4 = nn.Conv2d(in_channels=8,out_channels=32,kernel_size=1,stride=1)

        self.head = DAHead_backbone64(in_channels=64,nclass=64)
        self.t_conv3d = nn.Conv3d(in_channels=2, out_channels=16, kernel_size=(3, 1, 1), stride=(2, 1, 1),
                        padding=(2, 0, 0))
        self.c_conv3d = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(2, 1, 1), stride=(1, 1, 1),
                        padding=(0, 0, 0))

        self.t_maxpool = nn.MaxPool3d(kernel_size=(2, 1, 1))
        self.avgpool = nn.AdaptiveAvgPool3d((1,128,128))

    def forward(self, x):
        c = x
        
        c = self.t_conv3d(c)                            #4,3,2,128,128
        c = self.relu(c)
        c = self.c_conv3d(c)
        c = self.relu(c)
        b = c
        c = self.t_maxpool(c)                       #4,3,1,128,128
        b = self.avgpool(b)
        c = c + b
        c = c.view(-1, 32, 128, 128)
        
        x = x.reshape(-1, 8, 128, 128)
        b = x
        x = self.conv1(x)           # (N, 16, win, r, a)
        # x = self.pool1(x)           # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.conv2(x)           # (N, 32, win/2, r/2, a/2)
        # x = self.pool2(x)           # (N, 32, win/4, r/4, a/4)
        x = self.relu(x)
        x = self.conv3(x) 
        x = self.relu(x)
        x = self.head(x)
        
        x = self.deconv1(x)         # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.deconv2(x)         # (N, n_class, win, r, a)
        x = self.relu(x)
        x = x + self.conv4(b)
        # x = self.upsample(x)
        # x0_np = x.cpu().detach().numpy()[0, 0, 0, :, :]
        # x1_np = x.cpu().detach().numpy()[0, 1, 0, :, :]
        # x2_np = x.cpu().detach().numpy()[0, 2, 0, :, :]
        # x = self.sigmoid(x)

        x = torch.cat((x,c),dim=1)#x = x + c
        return x  #b,64,128,128  

class BackboneNet_Dilation_V2(nn.Module):
    def __init__(self):
        super(BackboneNet_Dilation_V2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=0,dilation=2)
        self.relu = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=0,dilation=3)
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,dilation=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,stride=2,output_padding=1,dilation=3)
        self.conv4 = nn.Conv2d(in_channels=8,out_channels=32,kernel_size=1,stride=1)

        self.head = DAHead_backbone64(in_channels=64,nclass=64)
        self.conv3d = nn.Conv3d(in_channels=2,out_channels=16,kernel_size=(4,1,1),stride=1)
    def forward(self, x):
        c = x
        c = self.conv3d(c)
        c = c.view(-1, 16, 128, 128)
        x = x.reshape(-1, 8, 128, 128)
        b = x
        x = self.conv1(x)           # (N, 16, win, r, a)
        # x = self.pool1(x)           # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.conv2(x)           # (N, 32, win/2, r/2, a/2)
        # x = self.pool2(x)           # (N, 32, win/4, r/4, a/4)
        x = self.relu(x)
        x = self.conv3(x) 
        x = self.relu(x)
        x = self.head(x)
        
        x = self.deconv1(x)         # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.deconv2(x)         # (N, n_class, win, r, a)
        x = self.relu(x)
        x = x + self.conv4(b)
        # x = self.upsample(x)
        # x0_np = x.cpu().detach().numpy()[0, 0, 0, :, :]
        # x1_np = x.cpu().detach().numpy()[0, 1, 0, :, :]
        # x2_np = x.cpu().detach().numpy()[0, 2, 0, :, :]
        # x = self.sigmoid(x)

        x = torch.cat((x,c),dim=1)
        return x     #b,48,128,128


    
class BackboneNet_Dilation_V5(nn.Module):
    def __init__(self):
        super(BackboneNet_Dilation_V5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=0,dilation=2)
        self.relu = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=0,dilation=3)
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,dilation=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,stride=2,output_padding=1,dilation=3)
        self.conv4 = nn.Conv2d(in_channels=8,out_channels=32,kernel_size=1,stride=1)

        self.head = DAHead_backbone64(in_channels=64,nclass=64)
        self.conv3d = nn.Conv3d(in_channels=2,out_channels=16,kernel_size=(4,1,1),stride=1)
        self.channelattention = DA_channel_attention()
    def forward(self, x):
        c = x
        c = self.conv3d(c)
        c = c.view(-1, 16, 128, 128)
        c = self.channelattention(c)
        
        x = x.reshape(-1, 8, 128, 128)
        x = self.channelattention(x)
        b = x
        x = self.conv1(x)           # (N, 16, win, r, a)
        # x = self.pool1(x)           # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.conv2(x)           # (N, 32, win/2, r/2, a/2)
        # x = self.pool2(x)           # (N, 32, win/4, r/4, a/4)
        x = self.relu(x)
        x = self.conv3(x) 
        x = self.relu(x)
        x = self.head(x)
        
        x = self.deconv1(x)         # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.deconv2(x)         # (N, n_class, win, r, a)
        x = self.relu(x)
        x = x + self.conv4(b)
        # x = self.upsample(x)
        # x0_np = x.cpu().detach().numpy()[0, 0, 0, :, :]
        # x1_np = x.cpu().detach().numpy()[0, 1, 0, :, :]
        # x2_np = x.cpu().detach().numpy()[0, 2, 0, :, :]
        # x = self.sigmoid(x)

        x = torch.cat((x,c),dim=1)
        return x     #b,48,128,128
class BackboneNet_Head_V1_dilation(nn.Module):
    def __init__(self):
        super(BackboneNet_Head_V1_dilation, self).__init__()
        # self.conv0 = nn.Conv3d(2, 16, (6, 4, 4), stride=2, padding=(2, 1, 1),dilation=1)
        self.conv1 = nn.Conv3d(2, 32, (6, 4, 4), stride=2, padding=(2, 1, 1),dilation=1)
        # self.pool1 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, (2, 6, 6), stride=2, padding=(0, 1, 1),dilation=(1,2,2))
        # self.pool2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))


        self.conv5 = nn.Conv3d(2,16,(1,1,1),stride=(1,1,1),padding=(0,0,0))
        self.relu = nn.LeakyReLU()

        self.deconv2 = nn.ConvTranspose3d(64, 32, (2, 5, 5), stride=(2, 2, 2),dilation=(1,2,2),output_padding=(0,1,1))
        self.deconv3 = nn.ConvTranspose3d(32, 16, (2, 2, 2), stride=(2, 2, 2),dilation=1)


        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1))
        self.avg = nn.AdaptiveAvgPool3d((1,128,128))
        self.head = DAHead_backbone64(in_channels=64,nclass=64)
        self.conv6 = nn.Conv3d(16,32,(4,1,1),stride=(1,1,1),padding=(0,0,0))
    def forward(self, x):
        # batch_size, in_chirps, n_channels, w, h = x.shape
        b = x
        x = self.conv1(x)           # (N, 16, win, r, a)
        # x = self.pool1(x)           # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.conv2(x)           # (N, 32, win/2, r/2, a/2)
        # x = self.pool2(x)           # (N, 32, win/4, r/4, a/4)
        x = self.relu(x)
        x = x.view(-1, 64, 28, 28)
        x = self.head(x)
        x = x.view(-1, 64,1, 28, 28)
        # x = self.conv3(x)           # (N, 64, win/4, r/4, a/4)
        # # x = self.pool3(x)           # (N, 64, win/8, r/8, a/8)
        # x = self.relu(x)
        # x0_np = x.cpu().detach().numpy()[0, 0, 0, :, :]
        # x1_np = x.cpu().detach().numpy()[0, 1, 0, :, :]
        # x2_np = x.cpu().detach().numpy()[0, 2, 0, :, :]
                # x = self.deconv1(x)         # (N, 32, win/4, r/4, a/4)
        # x = self.relu(x)
        x = self.deconv2(x)         # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.deconv3(x)         # (N, n_class, win, r, a)
        x = self.relu(x)
        x = x + self.conv5(b)
        # x = self.upsample(x)
        # x0_np = x.cpu().detach().numpy()[0, 0, 0, :, :]
        # x1_np = x.cpu().detach().numpy()[0, 1, 0, :, :]
        # x2_np = x.cpu().detach().numpy()[0, 2, 0, :, :]
        # x = self.sigmoid(x)
        b = x
        c = x
        x = self.avg(x)
        b = self.maxpool(b)
        x = torch.cat((x,b),dim=1)
        x = x  + self.conv6(c)
        # x = self.sigmoid(x)
        x = x.view(-1, 32, 128, 128)
        return x  
class BackboneNet_MAX_V1(nn.Module):
    def __init__(self):
        super(BackboneNet_MAX_V1, self).__init__()
        self.conv0 = nn.Conv3d(2, 16, (1, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv1 = nn.Conv3d(16, 32, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool1 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        # self.conv3 = nn.Conv3d(32, 64, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool3 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        # self.conv4 = nn.Conv3d(64, 128, (6, 4, 4), stride=2, padding=(2, 1, 1))
        # self.pool4 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        # self.conv5 = nn.Conv3d(2,16,(1,1,1),stride=(1,1,1),padding=(0,0,0))
        self.relu = nn.PReLU()
        # self.deconv1 = nn.ConvTranspose3d(64, 32, (2, 2, 2), stride=(2, 2, 2))
        self.deconv2 = nn.ConvTranspose3d(64, 32, (2, 2, 2), stride=(2, 2, 2))
        self.deconv3 = nn.ConvTranspose3d(32, 16, (2, 2, 2), stride=(2, 2, 2))

        # self.sigmoid = nn.Sigmoid()
        # self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1))
        self.avg = nn.AdaptiveAvgPool3d((1,128,128))
        self.head = DAHead_backbone64(in_channels=64,nclass=64)
        self.conv6 = nn.Conv3d(16,32,(4,1,1),stride=(1,1,1),padding=(0,0,0))
    def forward(self, x):
        # batch_size, in_chirps, n_channels, w, h = x.shape
        
        x = self.conv0(x)
        b = x
        x = self.relu(x)        
        x = self.conv1(x)           # (N, 16, win, r, a)
        c = x
        # x = self.pool1(x)           # (N, 16, win/2, r/2, a/2)
        x = self.relu(x)
        x = self.conv2(x)           # (N, 32, win/2, r/2, a/2)
        # x = self.pool2(x)           # (N, 32, win/4, r/4, a/4)
        x = self.relu(x)
        x = x.view(-1, 64, 32, 32)
        x = self.head(x)
        x = x.view(-1, 64,1, 32, 32)
        # x = self.conv3(x)           # (N, 64, win/4, r/4, a/4)
        # # x = self.pool3(x)           # (N, 64, win/8, r/8, a/8)
        # x = self.relu(x)
        # x0_np = x.cpu().detach().numpy()[0, 0, 0, :, :]
        # x1_np = x.cpu().detach().numpy()[0, 1, 0, :, :]
        # x2_np = x.cpu().detach().numpy()[0, 2, 0, :, :]
                # x = self.deconv1(x)         # (N, 32, win/4, r/4, a/4)
        # x = self.relu(x)
        x = self.deconv2(x)         # (N, 16, win/2, r/2, a/2)
        x = x + c
        x = self.relu(x)
        x = self.deconv3(x)         # (N, n_class, win, r, a)
        
        x = x + b
        x = self.relu(x)
        # x = self.upsample(x)
        # x0_np = x.cpu().detach().numpy()[0, 0, 0, :, :]
        # x1_np = x.cpu().detach().numpy()[0, 1, 0, :, :]
        # x2_np = x.cpu().detach().numpy()[0, 2, 0, :, :]
        # x = self.sigmoid(x)
        b = x
        c = x
        x = self.avg(x)
        b = self.maxpool(b)
        x = torch.cat((x,b),dim=1)
        x = x  + self.conv6(c)
        x = self.relu(x)
        # x = self.sigmoid(x)
        x = x.view(-1, 32, 128, 128)
        return x 
def unsample1(x):
    return F.interpolate(x,scale_factor=2, mode='bilinear')
def unsample(x):
    return F.interpolate(x,scale_factor=2, mode='trilinear')
def unsample_out1(x):
    return F.interpolate(x,scale_factor=4, mode='bilinear')
def unsample_out2(x):
    return F.interpolate(x,scale_factor=2, mode='bilinear')    
class BackboneNet_MAX_V2(nn.Module):
    def __init__(self):
        super(BackboneNet_MAX_V2, self).__init__()
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

        self.relu = nn.PReLU()


        # self.deconv2 = F.interpolate(size=(2,64,64), scale_factor=2, mode='bilinear')
        # self.deconv3 = F.interpolate(input,size=(4,128,128), scale_factor=2, mode='bilinear')


        # self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1))
        self.avg = nn.AdaptiveAvgPool3d((1,128,128))
        self.head = DAHead_backbone64(in_channels=64,nclass=64)
        self.conv6 = nn.Conv3d(16,32,(4,1,1),stride=(1,1,1),padding=(0,0,0))
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
        x = self.relu(x)          #(N, 32, 2, 64, 64) 
        
        b = x
        c = x
        x = self.avg(x)
        b = self.maxpool(b)
        x = torch.cat((x,b),dim=1)
        x = x  + self.conv6(c)
        x = self.relu(x)
        # x = self.sigmoid(x)
        x = x.view(-1, 32, 128, 128)
        return x  
class BackboneNet_MAX_V3(nn.Module):
    def __init__(self):
        super(BackboneNet_MAX_V3, self).__init__()
        self.conv0 = nn.Conv3d(2, 16, (1, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv1 = nn.Conv3d(16, 32, (6, 4, 4), stride=2, padding=(2, 1, 1))

        self.conv_a = nn.Conv3d(16, 16, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_a1 = nn.Conv3d(16, 16, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_b = nn.Conv3d(32, 32, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_b1 = nn.Conv3d(32, 32, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_c = nn.Conv3d(64, 64, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_c1 = nn.Conv3d(64, 64, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_d = nn.Conv2d(64, 64, ( 3, 3), stride=1, padding=( 1, 1))
        self.conv_d1 = nn.Conv2d(64, 32, ( 3, 3), stride=1, padding=( 1, 1))
        self.conv_e = nn.Conv2d(32, 32, (3, 3), stride=1, padding=( 1, 1))
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
        x = x + c
        x = self.relu(x)          #(N, 32,  64, 64)  
        
        x = unsample1(x)         #(N, 32,  128, 128) 
        
        x = self.conv_e(x)
        x = self.relu(x)
        x = self.conv_e1(x)
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


class BackboneNet_MAX_V5(nn.Module):
    def __init__(self):
        super(BackboneNet_MAX_V5, self).__init__()
        self.conv0 = nn.Conv3d(2, 16, (1, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv1 = nn.Conv3d(16, 32, (6, 4, 4), stride=2, padding=(2, 1, 1))

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
        
        self.conv_d = nn.Conv2d(64, 64, ( 3, 3), stride=1, padding=( 1, 1))
        self.conv_d1 = nn.Conv2d(64, 32, ( 3, 3), stride=1, padding=( 1, 1))
        self.conv_e = nn.Conv2d(32, 32, (3, 3), stride=1, padding=( 1, 1))
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
        x = x + d1
        x = self.relu(x)          #(N, 64,  32, 32) 
        
        # x = x.view(-1, 64, 32, 32)
        
        
        # x = self.head(x)        # (N, 64, 32, 32)


        x = unsample1(x)                 # (N, 64, 64, 64)

        
        x = self.conv_d(x)
        x = self.relu(x)
        x = self.conv_d1(x)
        c = (self.maxpoolc(c)).view(-1, 32, 64, 64)
        x = x + c
        x = self.relu(x)          #(N, 32,  64, 64)  
        
        x = unsample1(x)         #(N, 32,  128, 128) 
        
        x = self.conv_e(x)
        x = self.relu(x)
        x = self.conv_e1(x)
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


class BackboneNet_MAX_V6(nn.Module):
    def __init__(self):
        super(BackboneNet_MAX_V6, self).__init__()
        self.conv0 = nn.Conv3d(2, 16, (1, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv1 = nn.Conv3d(16, 32, (6, 4, 4), stride=2, padding=(2, 1, 1))

        self.conv_a = nn.Conv3d(16, 16, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_a1 = nn.Conv3d(16, 16, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_a2 = nn.Conv3d(16, 16, (1, 3, 3), stride=1, padding=(0, 1, 1))
        
        self.conv_b = nn.Conv3d(32, 32, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_b1 = nn.Conv3d(32, 32, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_b2 = nn.Conv3d(32, 32, (1, 3, 3), stride=1, padding=(0, 1, 1))
        
        self.conv_c = nn.Conv3d(64, 64, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_c1 = nn.Conv3d(64, 64, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_c2 = nn.Conv3d(64, 64, (1, 3, 3), stride=1, padding=(0, 1, 1))
        
        self.conv_d = nn.Conv2d(64, 64, ( 3, 3), stride=1, padding=( 1, 1))
        self.conv_d1 = nn.Conv2d(64, 64, ( 3, 3), stride=1, padding=( 1, 1))
        self.conv_d2 = nn.Conv2d(64, 32, ( 3, 3), stride=1, padding=( 1, 1))
        
        self.conv_e = nn.Conv2d(32, 32, (3, 3), stride=1, padding=( 1, 1))
        self.conv_e1 = nn.Conv2d(32, 32, ( 3, 3), stride=1, padding=( 1, 1))
        self.conv_e2 = nn.Conv2d(32, 16, ( 3, 3), stride=1, padding=( 1, 1))
               
        self.conv2 = nn.Conv3d(32, 64, (6, 4, 4), stride=2, padding=(2, 1, 1))

        self.relu = nn.PReLU()

        self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1))
        self.maxpoolc = nn.MaxPool3d(kernel_size=(2, 1, 1))

        self.head = DAHead_backbone64(in_channels=64,nclass=64)
        self.conv6 = nn.Conv2d(32,32,( 3, 3), stride=1, padding=( 1, 1))
        self.conv7 = nn.Conv2d(32,32,( 3, 3), stride=1, padding=( 1, 1))
        self.conv7_1 = nn.Conv2d(32,32,( 3, 3), stride=1, padding=( 1, 1))
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
        
        x = self.conv2(x)           # (N, 64, 1, 32, 32)
        d = x
        
        x = self.conv_c(x)
        x = self.relu(x)
        x = self.conv_c1(x)
        x = self.relu(x)
        x = self.conv_c2(x)
        x = x + d
        x = self.relu(x)        # (N, 64, 1, 32, 32)
  
        x = x.view(-1, 64, 32, 32)
        x = self.head(x)        # (N, 64, 32, 32)


        x = unsample1(x)                 # (N, 64, 64, 64)

        
        x = self.conv_d(x)
        x = self.relu(x)
        x = self.conv_d1(x)
        x = self.relu(x)
        x = self.conv_d2(x)
        c = (self.maxpoolc(c)).view(-1, 32, 64, 64)
        x = x + c
        x = self.relu(x)          #(N, 32,  64, 64)  
        
        x = unsample1(x)         #(N, 32,  128, 128) 
        
        x = self.conv_e(x)
        x = self.relu(x)
        x = self.conv_e1(x)
        x = self.relu(x)
        x = self.conv_e2(x)
        b = (self.maxpool(b)).view(-1, 16, 128, 128)
        x = x + b
        x = self.relu(x)          #(N, 16, 128, 128) 
        
        x = self.convk(x)
        x = self.relu(x)        #(N, 32, 128, 128)
        
        b = x
        x = self.conv6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.relu(x)
        x = self.conv7_1(x)
        x = x + b
        x = self.relu(x)          #(N, 32, 128, 128) 
        
        x = self.conv8(x)       #(N, 16, 128, 128) 
        x = self.conv9(x)       #(N, 16, 128, 128)
        x = self.sigm(x)        #(N, 3, 128, 128)
        return x 


class BackboneNet_MAX_V7(nn.Module):
    def __init__(self):
        super(BackboneNet_MAX_V7, self).__init__()
        self.conv0 = nn.Conv3d(2, 16, (1, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv1 = nn.Conv3d(16, 32, (6, 4, 4), stride=2, padding=(2, 1, 1))

        self.conv_a = nn.Conv3d(16, 16, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_a1 = nn.Conv3d(16, 16, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_b = nn.Conv3d(32, 32, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_b1 = nn.Conv3d(32, 32, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_c = nn.Conv3d(64, 64, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_c1 = nn.Conv3d(64, 64, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_d = nn.Conv2d(64, 64, ( 3, 3), stride=1, padding=( 1, 1))
        self.conv_d1 = nn.Conv2d(64, 32, ( 3, 3), stride=1, padding=( 1, 1))
        self.conv_e = nn.Conv2d(32, 32, (3, 3), stride=1, padding=( 1, 1))
        self.conv_e1 = nn.Conv2d(32, 16, ( 3, 3), stride=1, padding=( 1, 1))
               
        self.conv2 = nn.Conv3d(32, 64, (6, 4, 4), stride=2, padding=(2, 1, 1))

        self.relu = nn.PReLU()

        self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1))
        self.maxpoolc = nn.MaxPool3d(kernel_size=(2, 1, 1))

        self.head = DAHead_backbone64(in_channels=64,nclass=64)
        self.conv6 = nn.Conv2d(32,32,( 3, 3), stride=1, padding=( 1, 1))
        self.conv7 = nn.Conv2d(32,32,( 3, 3), stride=1, padding=( 1, 1))
        self.conv8 = nn.Conv2d(32,16,( 3, 3), stride=1, padding=( 1, 1))
        self.conv9 = nn.Conv2d(16,3,( 1, 1), stride=1, padding=( 0, 0))
        self.convk = nn.Conv2d(16,32,( 3, 3), stride=1, padding=( 1, 1))
        self.sigm = nn.Sigmoid()
        self.side1 = nn.Conv2d(64,3,3,padding=1)
        self.side2 = nn.Conv2d(32,3,3,padding=1)
        self.outconv = nn.Conv2d(9,3,1)
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
        
        out1 = self.side1(x)
        out1 = unsample_out1(out1)
        x = unsample1(x)                 # (N, 64, 64, 64)

        
        x = self.conv_d(x)
        x = self.relu(x)
        x = self.conv_d1(x)
        c = (self.maxpoolc(c)).view(-1, 32, 64, 64)
        x = x + c
        x = self.relu(x)          #(N, 32,  64, 64)  
        
        out2 = self.side2(x)
        out2 = unsample_out2(out2)
        x = unsample1(x)         #(N, 32,  128, 128) 
        
        x = self.conv_e(x)
        x = self.relu(x)
        x = self.conv_e1(x)
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
        
        x = self.conv8(x)       #(N, 16, 128, 128) 
        x = self.conv9(x)       #(N, 3, 128, 128)
        x = self.outconv(torch.cat((out1,out2,x),1))
        x = self.sigm(x)        #(N, 3, 128, 128)
        return x, self.sigm(out1), self.sigm(out2) 


class BackboneNet_MAX_V8(nn.Module):
    def __init__(self):
        super(BackboneNet_MAX_V8, self).__init__()
        self.conv0 = nn.Conv3d(2, 16, (1, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv1 = nn.Conv3d(16, 32, (6, 4, 4), stride=2, padding=(2, 1, 1))

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
        
        self.conv_d = nn.Conv2d(64, 64, ( 3, 3), stride=1, padding=( 1, 1))
        self.conv_d1 = nn.Conv2d(64, 32, ( 3, 3), stride=1, padding=( 1, 1))
        self.conv_e = nn.Conv2d(32, 32, (3, 3), stride=1, padding=( 1, 1))
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
        self.conv8 = nn.Conv2d(32,16,( 3, 3), stride=1, padding=( 1, 1))
        self.conv9 = nn.Conv2d(16,3,( 1, 1), stride=1, padding=( 0, 0))
        self.convk = nn.Conv2d(16,32,( 3, 3), stride=1, padding=( 1, 1))
        self.sigm = nn.Sigmoid()
        self.res1 = nn.Conv2d(128,64,1)
        self.res2 = nn.Conv2d(64,32,1)
        self.res3 = nn.Conv2d(32,16,1)
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
        d1_1 = self.res1(x)
        x = self.conv_c4(x)
        x = self.relu(x)
        x = self.conv_c5(x)     #(N, 64, 32, 32)
        d1 = (self.maxpoolc(d)).view(-1, 64, 32, 32)
        d1 = self.head(d1)
        x = x + d1 +d1_1
        x = self.relu(x)          #(N, 64,  32, 32) 
        
        x = unsample1(x)                 # (N, 64, 64, 64)
        d1_1 = self.res2(x)
        x = self.conv_d(x)
        x = self.relu(x)
        x = self.conv_d1(x)
        c = (self.maxpoolc(c)).view(-1, 32, 64, 64)
        x = x + c +d1_1
        x = self.relu(x)          #(N, 32,  64, 64)  
        
        x = unsample1(x)         #(N, 32,  128, 128) 
        d1_1 = self.res3(x)
        x = self.conv_e(x)
        x = self.relu(x)
        x = self.conv_e1(x)
        b = (self.maxpool(b)).view(-1, 16, 128, 128)
        x = x + b +d1_1
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
