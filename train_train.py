import os
import time
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from cruw import CRUW

from rodnet.datasets.CRDataset import CRDataset
from rodnet.datasets.CRDatasetSM import CRDatasetSM
from rodnet.datasets.CRDataLoader import CRDataLoader
from rodnet.datasets.collate_functions import cr_collate
from rodnet.core.radar_processing import chirp_amp
from rodnet.utils.solve_dir import create_dir_for_new_model
from rodnet.utils.load_configs import load_configs_from_file, parse_cfgs, update_config_dict
from rodnet.utils.visualization import visualize_train_img
from test_eval import testval
from rodnet.models.losses.loss import Dice_loss,BinaryDiceLoss,SoftDiceLoss,FocalLoss,FocalLoss1
from data_aug import Aug_data,transition_range

def parse_args():
    parser = argparse.ArgumentParser(description='Train RODNet.')

    parser.add_argument('--config', type=str,default='/home/zdk/RODNet-master/configs/config_rodnet_cdcv2_win16_mnet.py',  help='configuration file path')
    parser.add_argument('--sensor_config', type=str, default='sensor_config_rod2021')
    parser.add_argument('--data_dir', type=str, default='/home/zdk/RODNet-master/data/pkl', help='directory to the prepared data')
    parser.add_argument('--log_dir', type=str, default='/home/zdk/RODNet-master/checkpoints/', help='directory to save trained model')
    parser.add_argument('--resume_from', type=str, default='', help='path to the trained model')#default='',
    parser.add_argument('--save_memory', action="store_true", help="use customized dataloader to save memory")
    parser.add_argument('--use_noise_channel', action="store_true", help="use noise channel or not")

    parser = parse_cfgs(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config_dict = load_configs_from_file(args.config)
    config_dict = update_config_dict(config_dict, args)  # update configs by args

    # dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'])
    dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'], sensor_config_name=args.sensor_config)
    radar_configs = dataset.sensor_cfg.radar_cfg
    range_grid = dataset.range_grid
    angle_grid = dataset.angle_grid

    model_cfg = config_dict['model_cfg']
    if model_cfg['type'] == 'CDCv2':
        
     
        from rodnet.models.M_UHRNet import RODNetUNet as RODNet
    
    else:
        raise NotImplementedError

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    train_model_path = args.log_dir

    # create / load models
    cp_path = None
    epoch_start = 0
    iter_start = 0
    if args.resume_from is not None and os.path.exists(args.resume_from):
        cp_path = args.resume_from
        model_dir, model_name = create_dir_for_new_model(model_cfg['name'], train_model_path)
    else:
        model_dir, model_name = create_dir_for_new_model(model_cfg['name'], train_model_path)

    train_viz_path = os.path.join(model_dir, 'train_viz')
    if not os.path.exists(train_viz_path):
        os.makedirs(train_viz_path)

    writer = SummaryWriter(model_dir)
    save_config_dict = {
        'args': vars(args),
        'config_dict': config_dict,
    }
    config_json_name = os.path.join(model_dir, 'config-' + time.strftime("%Y%m%d-%H%M%S") + '.json')
    with open(config_json_name, 'w') as fp:
        json.dump(save_config_dict, fp)
    train_log_name = os.path.join(model_dir, "train.log")
    with open(train_log_name, 'w'):
        pass

    n_class = dataset.object_cfg.n_class
    n_epoch = config_dict['train_cfg']['n_epoch']
    batch_size = config_dict['train_cfg']['batch_size']
    lr = config_dict['train_cfg']['lr']
    if 'stacked_num' in model_cfg:
        stacked_num = model_cfg['stacked_num']
    else:
        stacked_num = None

    print("Building dataloader ... (Mode: %s)" % ("save_memory" if args.save_memory else "normal"))

    if not args.save_memory:
        crdata_train = CRDataset(data_dir=args.data_dir, dataset=dataset, config_dict=config_dict, split='train',
                                 noise_channel=args.use_noise_channel)
        # seq_names = crdata_train.seq_names
        # index_mapping = crdata_train.index_mapping
        dataloader = DataLoader(crdata_train, batch_size, shuffle=True, num_workers=3, collate_fn=cr_collate)

        # crdata_valid = CRDataset(os.path.join(args.data_dir, 'data_details'),
        #                          os.path.join(args.data_dir, 'confmaps_gt'),
        #                          win_size=win_size, set_type='valid', stride=8)
        # seq_names_valid = crdata_valid.seq_names
        # index_mapping_valid = crdata_valid.index_mapping
        # dataloader_valid = DataLoader(crdata_valid, batch_size=batch_size, shuffle=True, num_workers=0)

    else:
        crdata_train = CRDatasetSM(data_dir=args.data_dir, dataset=dataset, config_dict=config_dict, split='train',
                                   noise_channel=args.use_noise_channel)
        # seq_names = crdata_train.seq_names
        # index_mapping = crdata_train.index_mapping
        dataloader = CRDataLoader(crdata_train, shuffle=False, noise_channel=args.use_noise_channel)

        # crdata_valid = CRDatasetSM(os.path.join(args.data_dir, 'data_details'),
        #                          os.path.join(args.data_dir, 'confmaps_gt'),
        #                          win_size=win_size, set_type='train', stride=8, is_Memory_Limit=True)
        # seq_names_valid = crdata_valid.seq_names
        # index_mapping_valid = crdata_valid.index_mapping
        # dataloader_valid = CRDataLoader(crdata_valid, batch_size=batch_size, shuffle=True)

    if args.use_noise_channel:
        n_class_train = n_class + 1
    else:
        n_class_train = n_class

    print("Building model ... (%s)" % model_cfg)
    
   
    if model_cfg['type'] == 'CDCv2':
        # in_chirps = len(radar_configs['chirp_ids'])
        # rodnet = RODNet(num_classes=3,  mnet_cfg=(4,3),backbone = "UHRNet_W18_Small").cuda()
        rodnet = RODNet(mnet_cfg=(4,3)).cuda()
        focal_loss = FocalLoss()
        criterion0 = nn.BCELoss()
        criterion = nn.BCELoss()
        dice_loss = Dice_loss()
  
        mse = nn.L1Loss()
        # binary_loss =BinaryDiceLoss()
    else:
        raise TypeError
    optimizer = optim.Adam(rodnet.parameters(), lr=lr)  #,weight_decay=0.00005
    # optimizer = optim.Adam(rodnet.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # scheduler = StepLR(optimizer, step_size=config_dict['train_cfg']['lr_step'], gamma=0.1)
    scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=6978, step_size_down=6978,
                        cycle_momentum=False)                               #6978(b=32) 13962(b=16) 9306(b=24) 3486(b=64) 27930(b=8)18618(b=12)
    best_ap_toal = 0
    lowest_loss_test = 10
    iter_count = 0
    loss_ave = 0
    loss_weight = 1
    IS_Augdata = True
    IS_MixAug = True
    data_va = None
    if cp_path is not None:
        checkpoint = torch.load(cp_path)
        if 'optimizer_state_dict' in checkpoint:
            rodnet.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_start = checkpoint['epoch'] + 1
            iter_start = checkpoint['iter'] + 1
            loss_cp = checkpoint['loss']
            if 'iter_count' in checkpoint:
                iter_count = checkpoint['iter_count']
            if 'loss_ave' in checkpoint:
                loss_ave = checkpoint['loss_ave']
        else:
            rodnet.load_state_dict(checkpoint['model_state_dict'])

    # print training configurations
    print("Model name: %s" % model_name)
    print("Number of sequences to train: %d" % crdata_train.n_seq)
    print("Training dataset length: %d" % len(crdata_train))
    print("Batch size: %d" % batch_size)
    print("Number of iterations in each epoch: %d" % int(len(crdata_train) / batch_size))

    for epoch in range(epoch_start, n_epoch):

        tic_load = time.time()
        if epoch == epoch_start:
            dataloader_start = iter_start
        else:
            dataloader_start = 0
        rodnet = rodnet.train()
        for iter, data_dict in enumerate(dataloader):
            
            data = data_dict['radar_data']
            data = np.transpose(data, (0, 2, 1, 3, 4))
            image_paths = data_dict['image_paths']
            confmap_gt = data_dict['anno']['confmaps']
            confmap_gt1 = confmap_gt.ge(0.1).float()

            if not data_dict['status']:
                # in case load npy fail
                print("Warning: Loading NPY data failed! Skip this iteration")
                tic_load = time.time()
                continue
                ########################################################################
            data_rv=None 
            data_va=None
            if IS_Augdata:
                if IS_MixAug:
                    data,  confmap_gt = Aug_data(data, data_rv, data_va, confmap_gt, type='mix')
                else:
                    data,  confmap_gt = Aug_data(data, data_rv, data_va, confmap_gt)
            ################################################
            tic = time.time()
            optimizer.zero_grad()  # zero the parameter gradients
            confmap_preds,_, _, _, _  = rodnet(data.float().cuda())               #,d1, d2, d3, d6 
            confmap_preds1 = confmap_preds.ge(0.1).float().cuda()
            loss_confmap = 0
            loss_1 = criterion0(confmap_preds1, confmap_gt1.float().cuda())
            loss_2 = criterion(confmap_preds, confmap_gt.float().cuda())
            loss_3 = focal_loss(confmap_preds, confmap_gt.float().cuda())
            # loss_4 = dice_loss(confmap_preds, confmap_gt.float().cuda())
            loss_confmap = loss_1 + loss_2 + loss_3 #+ loss_4  #loss_weight*(mse(confmap_preds, confmap_gt.float().cuda())) #+0.2*criterion(d1, confmap_gt.float().cuda())+0.6*criterion(d2, confmap_gt.float().cuda())
            # loss_confmap = criterion(confmap_preds, confmap_gt.float().cuda())+  loss_weight*(binary_loss(confmap_preds, confmap_gt.float().cuda()))  +criterion(confmap_preds, confmap_gt.float().cuda())
            #focal_loss(confmap_preds, confmap_gt.cuda()) +
            loss_confmap.backward()
            optimizer.step()
            tic_back = time.time()

            loss_ave = np.average([loss_ave, loss_confmap.item()], weights=[iter_count, 1])

            if iter % config_dict['train_cfg']['log_step'] == 0:
                # print statistics
                load_time = tic - tic_load
                back_time = tic_back - tic
                print('epoch %2d, iter %4d: loss: %.4f (%.4f) loss_bce: %.4f loss_confidence: %.4f loss_focal: %.4f loss_dice: %.4f| load time: %.2f | back time: %.2f' %
                      (epoch + 1, iter + 1, loss_confmap.item(), loss_ave, loss_confmap.item(),loss_confmap.item(),loss_confmap.item(), loss_confmap.item(),load_time, back_time))
                with open(train_log_name, 'a+') as f_log:
                    f_log.write('epoch %2d, iter %4d: loss: %.4f (%.4f) loss_bce: %.4f loss_confidence: %.4f loss_focal: %.4f loss_dice: %.4f| load time: %.2f | back time: %.2f\n' %
                                (epoch + 1, iter + 1, loss_confmap.item(), loss_ave,loss_confmap.item(),loss_confmap.item(),loss_confmap.item(), loss_confmap.item(),load_time, back_time))

                writer.add_scalar('loss/loss_all', loss_confmap.item(), iter_count)
                writer.add_scalar('loss/loss_ave', loss_ave, iter_count)
                writer.add_scalar('time/time_load', load_time, iter_count)
                writer.add_scalar('time/time_back', back_time, iter_count)
                # writer.add_scalar('param/param_lr', scheduler.get_last_lr()[0], iter_count)
                # writer.add_scalar('param/param_lr', scheduler.get_lr()[-1], epoch + 1)
                
                writer.add_scalar('param/param_lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch + 1)
                # if stacked_num is not None:
                #     confmap_pred = confmap_preds[stacked_num - 1].cpu().detach().numpy()
                # else:
                # soft = nn.Softmax()
                # confmap_preds = soft(confmap_preds)                
                confmap_pred = confmap_preds.cpu().detach().numpy()

               
                chirp_amp_curr = chirp_amp(data.numpy()[0, :, 2,  :, :], radar_configs['data_type'])


                
                fig_name = os.path.join(train_viz_path,
                                        '%03d_%010d_%06d.png' % (epoch + 1, iter_count, iter + 1))
                img_path = image_paths[0][0]
                visualize_train_img(fig_name, img_path, chirp_amp_curr,
                                    confmap_pred[0, :n_class, :, :],
                                    confmap_gt[0, :n_class, :, :])

            # if (iter + 1) % config_dict['train_cfg']['save_step'] == 0:
            #     # validate current model
            #     # print("validing current model ...")
            #     # validate()

            #     # save current model
            #     print("saving current model ...")
            #     status_dict = {
            #         'model_name': model_name,
            #         'epoch': epoch + 1,
            #         'iter': iter + 1,
            #         'model_state_dict': rodnet.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'loss': loss_confmap.item(),
            #         'loss_ave': loss_ave,
            #         'iter_count': iter_count,
            #     }
            #     save_model_path = '%s/epoch_%02d_iter_%010d.pkl' % (model_dir, epoch + 1, iter_count + 1)
            #     torch.save(status_dict, save_model_path)

            iter_count += 1
            tic_load = time.time()
            scheduler.step()
            
        # save current model
        print("saving current epoch model ...")
        status_dict = {
            'model_name': model_name,
            'epoch': epoch,
            'iter': iter,
            'model_state_dict': rodnet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_confmap.item(),
            'loss_ave': loss_ave,
            'iter_count': iter_count,
        }
        # save_model_path = '%s/epoch_%02d_final.pkl' % (model_dir, epoch + 1)
        save_model_path = '%s/epoch_final.pkl' % (model_dir)
        torch.save(status_dict, save_model_path)

        # scheduler.step()
        if (epoch+1) % 1 == 0:
            # from tools.test_eval import testval
            ap_toal,ar_toal,loss_test = testval(test_model=rodnet.eval(),model_name=status_dict['model_name'])
            with open(train_log_name, 'a+') as f_log:
                    f_log.write('epoch %2d, iter %4d: AP: %.4f | AR: %.4f | LOSS: %.4f\n' %
                                (epoch + 1, iter + 1, ap_toal, ar_toal, loss_test))

            writer.add_scalar('AP/ap_toal', ap_toal, epoch + 1)
            writer.add_scalar('AR/ar_toal', ar_toal, epoch + 1)
            writer.add_scalar('LOSS/loss_test', loss_test, epoch + 1)
            if best_ap_toal < ap_toal:
                best_ap_toal = ap_toal
                save_model_path = '%s/epoch_best.pkl' % (model_dir)
                torch.save(status_dict, save_model_path)
            if loss_test < lowest_loss_test:
                lowest_loss_test = loss_test
                save_model_path = '%s/epoch_lowest.pkl' % (model_dir)
                torch.save(status_dict, save_model_path)
            
        else:
            None
        
    print('Training Finished.')
