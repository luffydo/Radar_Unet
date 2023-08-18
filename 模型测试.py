from rodnet.models.M_UHRNet import RODNetUNet as RODNet
import torch
from torchsummary import summary

from cruw import CRUW
from rodnet.utils.load_configs import load_configs_from_file, parse_cfgs, update_config_dict
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train RODNet.')

    parser.add_argument('--config', type=str,default='/media/zdk/SYQ/Radar_detection/RODNet-master/configs/config_rodnet_cdcv2_win16_mnet.py',  help='configuration file path')
    parser.add_argument('--sensor_config', type=str, default='sensor_config_rod2021')
    parser.add_argument('--data_dir', type=str, default='/media/zdk/SYQ/Radar_detection/RODNet-master/data/pkl', help='directory to the prepared data')
    parser.add_argument('--log_dir', type=str, default='/media/zdk/SYQ/Radar_detection/RODNet-master/checkpoints/', help='directory to save trained model')
    parser.add_argument('--resume_from', type=str, default=None, help='path to the trained model')
    parser.add_argument('--save_memory', action="store_true", help="use customized dataloader to save memory")
    parser.add_argument('--use_noise_channel', action="store_true", help="use noise channel or not")

    parser = parse_cfgs(parser)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # args = parse_args()
    # config_dict = load_configs_from_file(args.config)
    # config_dict = update_config_dict(config_dict, args)  # update configs by args
    # config_dict = load_configs_from_file('/media/zdk/SYQ/Radar_detection/RODNet-master/configs/config_rodnet_cdcv2_win16_mnet.py')   
    # dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'], sensor_config_name=args.sensor_config)
    # n_class = dataset.object_cfg.n_class
    # radar_configs = dataset.sensor_cfg.radar_cfg
    # in_chirps = len(radar_configs['chirp_ids'])
    
    # rodnet = RODNet(in_channels=in_chirps, n_class=n_class,
    #             #    mnet_cfg=config_dict['model_cfg']['mnet_cfg'],
    #                         ).cuda()
    # rodnet = RODNet().cuda()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # vgg = rodnet.to(device)
    
    # summary(vgg, input_size=(2,4, 128, 128),batch_size=1)
    
    # from torchstat import stat

    model =RODNet()
    # stat(model, (2,4, 128, 128))
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total/1e6))
    
