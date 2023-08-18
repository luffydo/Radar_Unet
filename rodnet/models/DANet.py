import torch
import torch.nn as nn


class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        # feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        # feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        # attention_s = self.softmax(torch.bmm(self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1), self.conv_c(x).view(batch_size, -1, height * width)))
        # feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(self.conv_d(x).view(batch_size, -1, height * width), (self.softmax(torch.bmm(self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1), self.conv_c(x).view(batch_size, -1, height * width)))).permute(0, 2, 1)).view(batch_size, -1, height, width)
        feat_e = self.alpha * feat_e + x

        return feat_e


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # batch_size, _, height, width = x.size()
        # feat_a = x.view(batch_size, -1, height * width)
        # feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        # attention = torch.bmm(feat_a, feat_a_transpose)
        # attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        # attention = self.softmax(attention_new)

        # feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        # out = self.beta * feat_e + x
        batch_size, _, height, width = x.size()
        # feat_a = x.view(batch_size, -1, height * width)
        # feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm( x.view(batch_size, -1, height * width), x.view(batch_size, -1, height * width).permute(0, 2, 1))
        attention = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention)

        attention = torch.bmm(attention, x.view(batch_size, -1, height * width)).view(batch_size, -1, height, width)
        attention = self.beta * attention + x
        return attention


class DAHead(nn.Module):
    def __init__(self, in_channels, nclass, aux=True, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(DAHead, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(inter_channels, nclass, 1),
            nn.Sigmoid()
        )
        
        if aux:
            self.conv_p3 = nn.Sequential(
                nn.Dropout(0.2),
                nn.Conv2d(inter_channels, nclass, 1),
                nn.Sigmoid()
            )
            self.conv_c3 = nn.Sequential(
                nn.Dropout(0.2),
                nn.Conv2d(inter_channels, nclass, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        # feat_p = self.conv_p1(x)
        # feat_p = self.pam(feat_p)
        # feat_p = self.conv_p2(feat_p)

        # feat_c = self.conv_c1(x)
        # feat_c = self.cam(feat_c)
        # feat_c = self.conv_c2(feat_c)

        # feat_fusion = feat_p + feat_c

        # outputs = []
        # fusion_out = self.out(feat_fusion)
        # outputs.append(fusion_out)
        # if self.aux:
        #     p_out = self.conv_p3(feat_p)
        #     c_out = self.conv_c3(feat_c)
        #     outputs.append(p_out)
        #     outputs.append(c_out)


        feat_p = self.conv_p2(self.pam(self.conv_p1(x)))


        feat_c = self.conv_c2(self.cam(self.conv_c1(x)))

        feat_fusion = feat_p + feat_c

        outputs = []
        feat_fusion = self.out(feat_fusion)
        
        outputs.append(feat_fusion)
        if self.aux:

            outputs.append(self.conv_p3(feat_p))
            outputs.append(self.conv_c3(feat_c))
        return tuple(outputs)                 #输出的是一个列表[0][1][2]，可以分别计算损失
class DAHead_backbone(nn.Module):
    def __init__(self, in_channels,   norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(DAHead_backbone, self).__init__()

        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, 32, 3, padding=1, bias=False),
            norm_layer(32, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, 32, 3, padding=1, bias=False),
            norm_layer(32, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )



    def forward(self, x):
        # feat_p = self.conv_p1(x)
        # feat_p = self.pam(feat_p)
        # feat_p = self.conv_p2(feat_p)

        # feat_c = self.conv_c1(x)
        # feat_c = self.cam(feat_c)
        # feat_c = self.conv_c2(feat_c)

        # feat_fusion = feat_p + feat_c

        # outputs = []
        # fusion_out = self.out(feat_fusion)
        # outputs.append(fusion_out)
        # if self.aux:
        #     p_out = self.conv_p3(feat_p)
        #     c_out = self.conv_c3(feat_c)
        #     outputs.append(p_out)
        #     outputs.append(c_out)




        feat_fusion = self.conv_p2(self.pam(self.conv_p1(x))) + self.conv_c2(self.cam(self.conv_c1(x)))

  
        

        return feat_fusion 
class DAHead_backbone64(nn.Module):
    def __init__(self, in_channels,   norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(DAHead_backbone64, self).__init__()

        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, 64, 3, padding=1, bias=False),
            norm_layer(64, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, 64, 3, padding=1, bias=False),
            norm_layer(64, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )



    def forward(self, x):
    




        feat_fusion = self.conv_p2(self.pam(self.conv_p1(x))) + self.conv_c2(self.cam(self.conv_c1(x)))

  
        

        return feat_fusion 
class DAHead_backbone128(nn.Module):
    def __init__(self, in_channels,   norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(DAHead_backbone128, self).__init__()

        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, 128, 3, padding=1, bias=False),
            norm_layer(128, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, 128, 3, padding=1, bias=False),
            norm_layer(128, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )



    def forward(self, x):
    




        feat_fusion = self.conv_p2(self.pam(self.conv_p1(x))) + self.conv_c2(self.cam(self.conv_c1(x)))

  
        

        return feat_fusion 

class DAHead_backbone192(nn.Module):
    def __init__(self, in_channels,   norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(DAHead_backbone192, self).__init__()

        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, 192, 3, padding=1, bias=False),
            norm_layer(192, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, 192, 3, padding=1, bias=False),
            norm_layer(192, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )



    def forward(self, x):
    




        feat_fusion = self.conv_p2(self.pam(self.conv_p1(x))) + self.conv_c2(self.cam(self.conv_c1(x)))

  
        

        return feat_fusion 
class DAHead_backbone256(nn.Module):
    def __init__(self, in_channels,   norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(DAHead_backbone256, self).__init__()

        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, 256, 3, padding=1, bias=False),
            norm_layer(256, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, 256, 3, padding=1, bias=False),
            norm_layer(256, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )



    def forward(self, x):
    




        feat_fusion = self.conv_p2(self.pam(self.conv_p1(x))) + self.conv_c2(self.cam(self.conv_c1(x)))

  
        

        return feat_fusion
class DAHead_backbone384(nn.Module):
    def __init__(self, in_channels,   norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(DAHead_backbone384, self).__init__()

        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, 384, 3, padding=1, bias=False),
            norm_layer(384, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, 384, 3, padding=1, bias=False),
            norm_layer(384, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )



    def forward(self, x):
    




        feat_fusion = self.conv_p2(self.pam(self.conv_p1(x))) + self.conv_c2(self.cam(self.conv_c1(x)))

  
        

        return feat_fusion 
class DAHead_Position(nn.Module):
    def __init__(self, in_channels,   norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(DAHead_Position, self).__init__()

        inter_channels = in_channels // 4

        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, 3, padding=1, bias=False),
            norm_layer(in_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )


    def forward(self, x):
    




        feat_fusion = self.conv_p2(self.pam(self.conv_p1(x)))

  
        

        return feat_fusion  

class DAHead_Channel(nn.Module):
    def __init__(self, in_channels,   norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(DAHead_Channel, self).__init__()

        inter_channels = in_channels // 4

        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

        self.cam = _ChannelAttentionModule(**kwargs)

        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, 3, padding=1, bias=False),
            norm_layer(in_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )



    def forward(self, x):
    




        feat_fusion =  self.conv_c2(self.cam(self.conv_c1(x)))

  
        

        return feat_fusion 
class DA_channel_attention(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(DA_channel_attention, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # batch_size, _, height, width = x.size()
        # feat_a = x.view(batch_size, -1, height * width)
        # feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        # attention = torch.bmm(feat_a, feat_a_transpose)
        # attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        # attention = self.softmax(attention_new)

        # feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        # out = self.beta * feat_e + x
        batch_size, _, height, width = x.size()
        # feat_a = x.view(batch_size, -1, height * width)
        # feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm( x.view(batch_size, -1, height * width), x.view(batch_size, -1, height * width).permute(0, 2, 1))
        attention = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention)

        attention = torch.bmm(attention, x.view(batch_size, -1, height * width)).view(batch_size, -1, height, width)
        attention = self.beta * attention + x
        return attention      