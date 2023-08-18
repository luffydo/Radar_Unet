#--------------------------------------------------------------------#
#作用：计算模型需要占用的显存，方便知道显卡够不够用
#使用方法：将模型初始化之后，传入Calculate_gpu_memory()即可
#--------------------------------------------------------------------#

import torch
import numpy as np
import torchvision
import torch.nn as nn
from rodnet.models.M_UHRNet import RODNetUNet as RODNet
def Calculate_gpu_memory(Model,train_batch_size,img_wide,img_height):
    print("----------------计算模型要占用的显存------------")
    #step1#------------------------------------------------------------------计算模型参数占用的显存
    type_size = 4 #因为参数是float32,也就是4B
    para = sum([np.prod(list(p.size())) for p in Model.parameters()])
    print("Model {}:params:{:4f}M".format(Model._get_name(),para * type_size/1000/1000))
    #step2#------------------------------------------------------------------------计算模型的中间变量会占用的显存
    input = torch.ones((train_batch_size,4,2,img_wide, img_height)).cuda()
    input.requires_grad_(requires_grad=False)
    #遍历模型的每一个网络层（注意：一般模型都是嵌套建立的，这里只考虑了小于等于2层嵌套结构）
    mods = list(Model.named_children())
    out_sizes = []
    for i in range(0, len(mods)):
            mod = list(mods[i][1].named_children())
            if mod != []:
                for j in range(0, len(mod)):
                    m = mod[j][1]
                    #注意这里，如果relu激活函数是inplace则不用计算
                    if isinstance(m,nn.ReLU):  
                        if m.inplace:
                            continue
                    print("网络层(不包括池化层,inplace为True的激活函数)：",m)
                    try: #一般不会把展平操作记录到里面去，因为没有在__init__中初始化，所以这里需要加上，如果不加上，将不能继续计算
                        out = m(input)
                    except RuntimeError:
                        input = torch.flatten(input, 1)
                        out = m(input)
                    out_sizes.append(np.array(out.size()))
                    if mod[j][0] not in ["rpn_score","rpn_loc"]: 
                        input = out
            else:
                m = mods[i][1]
                #注意这里，如果relu激活函数是inplace则不用计算
                if isinstance(m,nn.ReLU):  
                    if m.inplace:
                        continue
                print("网络层(不包括池化层,inplace为True的激活函数)：",m)
                try:
                    out = m(input)
                except RuntimeError:
                    input = torch.flatten(input, 1)
                    out = m(input)
                out_sizes.append(np.array(out.size()))

                if mods[j][0] not in ["rpn_score","rpn_loc"]:
                    input = out
    #统计每一层网络中间变量需要占用的显存
    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums
    print('Model {} : intermedite variables: {:3f} M (without backward)'
            .format(Model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
            .format(Model._get_name(), total_nums * type_size*2 / 1000 / 1000))
    print("----------------显存计算完毕------------")


#------------------------------------------------------------------------#
#测试，下面的代码不会影响上面的函数被其他python文件导入
if __name__=="__main__":
    rodnet = RODNet(num_classes=3,  mnet_cfg=(4,3),backbone = "UHRNet_W18_Small").cuda()  #.cuda()
    print(rodnet)
    Calculate_gpu_memory(rodnet,16,128,128)

# python tools/train.py --config configs/config_rodnet_cdcv2_win16_mnet.py 