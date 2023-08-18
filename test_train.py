import os
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from cruw import CRUW

from rodnet.datasets.CRDataset import CRDataset,TESTDataset
from rodnet.datasets.collate_functions import cr_collate
from rodnet.core.post_processing import post_process, post_process_single_frame
from rodnet.core.post_processing import write_dets_results, write_dets_results_single_frame
from rodnet.core.post_processing import ConfmapStack
from rodnet.core.radar_processing import chirp_amp
from rodnet.utils.visualization import visualize_test_img, visualize_test_img_wo_gt
from rodnet.utils.load_configs import load_configs_from_file, parse_cfgs, update_config_dict
from rodnet.utils.solve_dir import create_random_model_name
from rodnet.models.M_UHRNet import RODNetUNet as RODNet

def parse_args():
    parser = argparse.ArgumentParser(description='Test RODNet.')

    # parser.add_argument('--config', type=str, help='choose rodnet model configurations')
    parser.add_argument('--config', type=str,default='/home/zdk/RODNet-master/configs/config_rodnet_cdcv2_win16_mnet.py',  help='configuration file path')
    parser.add_argument('--sensor_config', type=str, default='sensor_config_rod2021')
    parser.add_argument('--data_dir', type=str, default='/home/zdk/RODNet-master/data/pkl', help='directory to the prepared data')
    parser.add_argument('--checkpoint', default='/home/zdk/RODNet-master/checkpoints/144-MultiResUNet-Aug-attention-multi-cycli-bce+confidence(max_det15,1e-5)-20230310-143754/epoch_best.pkl', type=str, help='path to the saved trained model')
    parser.add_argument('--res_dir', type=str, default='/home/zdk/RODNet-master/results', help='directory to save testing results')
    parser.add_argument('--use_noise_channel', action="store_true", help="use noise channel or not")
    parser.add_argument('--demo', action="store_true", help='False: test with GT, True: demo without GT')
    parser.add_argument('--symbol', action="store_true", help='use symbol or text+score')

    parser = parse_cfgs(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    sybl = args.symbol

    config_dict = load_configs_from_file(args.config)
    config_dict = update_config_dict(config_dict, args)  # update configs by args

    dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'], sensor_config_name=args.sensor_config)
    radar_configs = dataset.sensor_cfg.radar_cfg
    range_grid = dataset.range_grid
    angle_grid = dataset.angle_grid

    model_cfg = config_dict['model_cfg']

    # if model_cfg['type'] == 'CDC':
    #     from rodnet.models import RODNetCDC as RODNet
    # elif model_cfg['type'] == 'HG':
    #     from rodnet.models import RODNetHG as RODNet
    # elif model_cfg['type'] == 'HGwI':
    #     from rodnet.models import RODNetHGwI as RODNet
    # elif model_cfg['type'] == 'CDCv2':
    #     from rodnet.models import RODNetCDCDCN as RODNet
    # elif model_cfg['type'] == 'HGv2':
    #     from rodnet.models import RODNetHGDCN as RODNet
    # elif model_cfg['type'] == 'HGwIv2':
    #     from rodnet.models import RODNetHGwIDCN as RODNet
    # else:
    #     raise NotImplementedError

    # parameter settings
    dataset_configs = config_dict['dataset_cfg']
    train_configs = config_dict['train_cfg']
    test_configs = config_dict['test_cfg']

    win_size = train_configs['win_size']
    n_class = dataset.object_cfg.n_class

    confmap_shape = (n_class, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])
    # if 'stacked_num' in model_cfg:
    #     stacked_num = model_cfg['stacked_num']
    # else:
    #     stacked_num = None
   
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        checkpoint_path = args.checkpoint
    else:
        raise ValueError("No trained model found.")

    if args.use_noise_channel:
        n_class_test = n_class + 1
    else:
        n_class_test = n_class

    print("Building model ... (%s)" % model_cfg)
    if model_cfg['type'] == 'CDC':
        rodnet = RODNet(in_channels=2, n_class=n_class_test).cuda()
    # elif model_cfg['type'] == 'HG':
    #     rodnet = RODNet(in_channels=2, n_class=n_class_test, stacked_num=stacked_num).cuda()
    # elif model_cfg['type'] == 'HGwI':
    #     rodnet = RODNet(in_channels=2, n_class=n_class_test, stacked_num=stacked_num).cuda()
    elif model_cfg['type'] == 'CDCv2':
        in_chirps = len(radar_configs['chirp_ids'])
        rodnet = RODNet(num_classes=3,  mnet_cfg=(4,3),backbone = "UHRNet_W18_Small").cuda()
    # elif model_cfg['type'] == 'HGv2':
    #     in_chirps = len(radar_configs['chirp_ids'])
    #     rodnet = RODNet(in_channels=in_chirps, n_class=n_class_test, stacked_num=stacked_num,
    #                     mnet_cfg=config_dict['model_cfg']['mnet_cfg'],
    #                     dcn=config_dict['model_cfg']['dcn']).cuda()
    # elif model_cfg['type'] == 'HGwIv2':
    #     in_chirps = len(radar_configs['chirp_ids'])
    #     rodnet = RODNet(in_channels=in_chirps, n_class=n_class_test, stacked_num=stacked_num,
    #                     mnet_cfg=config_dict['model_cfg']['mnet_cfg'],
    #                     dcn=config_dict['model_cfg']['dcn']).cuda()
    else:
        raise TypeError

    checkpoint = torch.load(checkpoint_path)
    if 'optimizer_state_dict' in checkpoint:
        rodnet.load_state_dict(checkpoint['model_state_dict'])
    else:
        rodnet.load_state_dict(checkpoint)
    if 'model_name' in checkpoint:
        model_name = checkpoint['model_name']
    else:
        model_name = create_random_model_name(model_cfg['name'], checkpoint_path)
    rodnet.eval()

    test_res_dir = os.path.join(os.path.join(args.res_dir, model_name))
    if not os.path.exists(test_res_dir):
        os.makedirs(test_res_dir)

    # save current checkpoint path
    weight_log_path = os.path.join(test_res_dir, 'weight_name.txt')
    if os.path.exists(weight_log_path):
        with open(weight_log_path, 'a+') as f:
            f.write(checkpoint_path + '\n')
    else:
        with open(weight_log_path, 'w') as f:
            f.write(checkpoint_path + '\n')

    total_time = 0
    total_count = 0

    data_root = dataset_configs['data_root']
    if not args.demo:
        seq_names = sorted(os.listdir(os.path.join(data_root, dataset_configs['train']['subdir'])))
    else:
        # seq_names = sorted(os.listdir(os.path.join(data_root, dataset_configs['demo']['subdir'])))
        seq_names = sorted(os.listdir(os.path.join('/media/zdk/SYQ/Radar_detection/RODNet-master/sequences/train')))
    print(seq_names)

    for seq_name in seq_names:
        seq_res_dir = os.path.join(test_res_dir, seq_name)
        if not os.path.exists(seq_res_dir):
            os.makedirs(seq_res_dir)
        seq_res_viz_dir = os.path.join(seq_res_dir, 'rod_viz')
        if not os.path.exists(seq_res_viz_dir):
            os.makedirs(seq_res_viz_dir)
        f = open(os.path.join(seq_res_dir, 'rod_res.txt'), 'w')
        f.close()                                                                #创建结果路径，每个序列都有文件夹

    for subset in seq_names:
        print(subset)
        if not args.demo:
            crdata_test = CRDataset(data_dir=args.data_dir, dataset=dataset, config_dict=config_dict, split='train',
                                    noise_channel=args.use_noise_channel, subset=subset, is_random_chirp=False)
        else:
            crdata_test = CRDataset(data_dir=args.data_dir, dataset=dataset, config_dict=config_dict, split='val',
                                    noise_channel=args.use_noise_channel, subset=subset, is_random_chirp=False)
        print("Length of testing data: %d" % len(crdata_test))
        dataloader = DataLoader(crdata_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=cr_collate)

        # seq_names = crdata_test.seq_names
        # index_mapping = crdata_test.index_mapping

        # init_genConfmap = ConfmapStack(confmap_shape)
        # init_genConfmap = np.zeros(confmap_shape)
        # iter_ = init_genConfmap
        # for i in range(train_configs['win_size'] - 1):
        #     while iter_.next is not None:
        #         iter_ = iter_.next
        #     iter_.next = ConfmapStack(confmap_shape)

        load_tic = time.time()
        for iter, data_dict in enumerate(dataloader):
            load_time = time.time() - load_tic
            data = data_dict['radar_data']
            data = np.transpose(data, (0, 2, 1, 3, 4))
            try:
                image_paths = data_dict['image_paths'][0]
            except:
                print('warning: fail to load RGB images, will not visualize results')
                image_paths = None
            seq_name = data_dict['seq_names'][0]
            # if not args.demo:
            #     confmap_gt = data_dict['anno']['confmaps']
            #     obj_info = data_dict['anno']['obj_infos']
            # else:
            #     confmap_gt = None
            #     obj_info = None
            if not args.demo:
                # confmap_gt = None
                # obj_info = None
                confmap_gt = data_dict['anno']['confmaps']
                obj_info = data_dict['anno']['obj_infos']  
            else:
                
                confmap_gt = data_dict['anno']['confmaps']
                obj_info = data_dict['anno']['obj_infos']                    #   互换条件成立的内容
            save_path = os.path.join(test_res_dir, seq_name, 'rod_res.txt')

            start_frame_id = data_dict['start_frame'].item()
            end_frame_id = data_dict['end_frame'].item()

            tic = time.time()
            confmap_pred,d1, d2, d3, d6  = rodnet(data.float().cuda())  #d4,d5,
            # if stacked_num is not None:
            #     confmap_pred = confmap_pred[-1].cpu().detach().numpy()  # (1, 4, 32, 128, 128)
            # else:
            confmap_pred = confmap_pred.cpu().detach().numpy()

            # if args.use_noise_channel:
            #     confmap_pred = confmap_pred[:, :n_class, :, :, :]

            infer_time = time.time() - tic
            total_time += infer_time

            # iter_ = init_genConfmap
            # for i in range(confmap_pred.shape[2]):
            #     if iter_.next is None and i != confmap_pred.shape[2] - 1:
            #         iter_.next = ConfmapStack(confmap_shape)
            #     iter_.append(confmap_pred[0, :, i, :, :])
            #     iter_ = iter_.next

            process_tic = time.time()
            # for i in range(test_configs['test_stride']):
            total_count += 1
            # res_final = post_process_single_frame(init_genConfmap.confmap, dataset, config_dict)
            res_final = post_process_single_frame(confmap_pred[0,:3,:,:], dataset, config_dict)                       #是否要降维，把batch qudiao
            # cur_frame_id = start_frame_id + i
            write_dets_results_single_frame(res_final, start_frame_id, save_path, dataset)
            # confmap_pred_0 = init_genConfmap.confmap
            # res_final_0 = res_final
            if image_paths is not None and iter%1==0:
                img_path = image_paths[0]                               #将i变成0
                radar_input = chirp_amp(data.numpy()[0, :, 2,  :, :], radar_configs['data_type'])
                fig_name = os.path.join(test_res_dir, seq_name, 'rod_viz', '%010d.jpg' % (start_frame_id))
                if confmap_gt is not None:
                    # confmap_gt_0 = confmap_gt[0, :, i, :, :]
                    # confmap_gt_0 = confmap_gt[0, :, :, :]
                    visualize_test_img(fig_name, img_path, radar_input, confmap_pred[0,:3,:,:],confmap_gt[0, :3, :, :], res_final,          
                                        dataset, sybl=sybl)                                                           #res_final_0变成res_final，confmap_pred_0变成confmap_pred
                else:
                    visualize_test_img_wo_gt(fig_name, img_path, radar_input, confmap_pred[0,:3,:,:], res_final,
                                                dataset, sybl=sybl)                                                          #res_final_0变成res_final，，confmap_pred_0变成confmap_pred
            # init_genConfmap = init_genConfmap.next

            # if iter == len(dataloader) - 1:
            #     offset = test_configs['test_stride']
            #     cur_frame_id = start_frame_id + offset
            #     while init_genConfmap is not None:
            #         total_count += 1
            #         res_final = post_process_single_frame(init_genConfmap.confmap, dataset, config_dict)
            #         write_dets_results_single_frame(res_final, cur_frame_id, save_path, dataset)
            #         confmap_pred_0 = init_genConfmap.confmap
            #         res_final_0 = res_final
            #         if image_paths is not None:
            #             img_path = image_paths[offset]
            #             radar_input = chirp_amp(data.numpy()[0, :, offset, :, :], radar_configs['data_type'])
            #             fig_name = os.path.join(test_res_dir, seq_name, 'rod_viz', '%010d.jpg' % (cur_frame_id))
            #             if confmap_gt is not None:
            #                 confmap_gt_0 = confmap_gt[0, :, offset, :, :]
            #                 visualize_test_img(fig_name, img_path, radar_input, confmap_pred_0, confmap_gt_0,
            #                                    res_final_0,
            #                                    dataset, sybl=sybl)
            #             else:
            #                 visualize_test_img_wo_gt(fig_name, img_path, radar_input, confmap_pred_0, res_final_0,
            #                                          dataset, sybl=sybl)
            #         init_genConfmap = init_genConfmap.next
            #         offset += 1
            #         cur_frame_id += 1

            # if init_genConfmap is None:
                # init_genConfmap = ConfmapStack(confmap_shape)


            proc_time = time.time() - process_tic
            # print("Testing %s: frame %4d to %4d | Load time: %.4f | Inference time: %.4f | Process time: %.4f" %
            #       (seq_name, start_frame_id, end_frame_id, load_time, infer_time, proc_time))

            load_tic = time.time()

    print("ave time: %f" % (total_time / total_count))
    
        
    
    
    
    """
    
    
    
    
    from cruw.eval import evaluate_rodnet_seq
    from cruw.eval.rod.rod_eval_utils import accumulate, summarize

    from rodnet.utils.load_configs import load_configs_from_file

    olsThrs = np.around(np.linspace(0.5, 0.9, int(np.round((0.9 - 0.5) / 0.05) + 1), endpoint=True), decimals=2)
    recThrs = np.around(np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01) + 1), endpoint=True), decimals=2)


# def parse_args():
#     # parser = argparse.ArgumentParser(description='Evaluate RODNet.')
#     # parser.add_argument('--data_root', type=str, default='/media/zdk/SYQ/Radar_detection/RODNet-master', help='directory to the prepared data')
#     # parser.add_argument('--sensor_config', type=str, default='sensor_config_rod2021')
#     parser.add_argument('--gt_dir', type=str, default='/media/zdk/SYQ/Radar_detection/RODNet-master/results', help='directory to ground truth')
#     parser.add_argument('--res_dir', type=str, default='/media/zdk/SYQ/Radar_detection/RODNet-master/results', help='directory to save testing results')
#     args = parser.parse_args()
#     return args


# if __name__ == '__main__':
#     
#     Example:
#         python eval.py --config configs/<CONFIG_FILE> --res_dir results/<FOLDER_NAME>
#     


    seq_names = sorted(os.listdir('/media/zdk/SYQ/Radar_detection/RODNet-master/results/rodnet-unet-mnet-20221209-233131'))
    seq_names = [name for name in seq_names if '.' not in name]

    evalImgs_all = []
    n_frames_all = 0

    for seq_name in seq_names:
        gt_path = os.path.join('/media/zdk/SYQ/Radar_detection/RODNet-master/annotations/train', seq_name.upper() + '.txt')
        res_path = os.path.join('/media/zdk/SYQ/Radar_detection/RODNet-master/results/rodnet-unet-mnet-20221209-233131', seq_name, 'rod_res.txt')

        data_path = os.path.join(dataset.data_root, 'sequences', 'train', gt_path.split('/')[-1][:-4])
        n_frame = len(os.listdir(os.path.join(data_path, dataset.sensor_cfg.camera_cfg['image_folder'])))

        evalImgs = evaluate_rodnet_seq(res_path, gt_path, n_frame, dataset)
        eval = accumulate(evalImgs, n_frame, olsThrs, recThrs, dataset, log=False)
        stats = summarize(eval, olsThrs, recThrs, dataset, gl=False)
        print("%s | AP_total: %.4f | AR_total: %.4f" % (seq_name.upper(), stats[0] * 100, stats[1] * 100))

        n_frames_all += n_frame
        evalImgs_all.extend(evalImgs)

    eval = accumulate(evalImgs_all, n_frames_all, olsThrs, recThrs, dataset, log=False)
    stats = summarize(eval, olsThrs, recThrs, dataset, gl=False)
    print("%s | AP_total: %.4f | AR_total: %.4f" % ('Overall'.ljust(18), stats[0] * 100, stats[1] * 100))
    
    """
