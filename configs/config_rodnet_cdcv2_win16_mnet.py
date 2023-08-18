dataset_cfg = dict(
    dataset_name='ROD2021',
    base_root="/home/zdk/RODNet-master",
    data_root="/home/zdk/RODNet-master/sequences",
    anno_root="/home/zdk/RODNet-master/annotations",
    # base_root="/media/zdk/SYQ/Radar_detection/RODNet-master",
    # data_root="/media/zdk/SYQ/Radar_detection/RODNet-master/data/pkl",
    # anno_root="/media/zdk/SYQ/Radar_detection/RODNet-master/annotations/train",
    anno_ext='.txt',
    train=dict(
        subdir='train',
        # seqs=['2019_05_29_BCMS000','2019_05_29_MLMS006','2019_05_29_PCMS005','2019_09_29_ONRD002'],  # can choose from the subdir folder
    ),
    valid=dict(
        subdir='valid',
        seqs=[],
    ),
    test=dict(
        subdir='test',
        # seqs=['2019_05_28_CM1S013','2019_05_28_MLMS005','2019_05_28_PBMS006'],  # can choose from the subdir folder
    ),
    demo=dict(
        subdir='train',
        seqs=[],
    ),
)

model_cfg = dict(
    # type='CDCv2',
    type='CDCv2',
    # name='rodnet-cdcv2-win16-mnet','123-MultiResUNet1-Aug-cycli-focal+bce+mse(max_det15,1e-5)''124-MultiResUNet1-Aug-cycli-focal+bce(max_det15,1e-5)'
    name='157-MultiNEW-Aug-attention-multi(attention)-cycli-newloss(max_det15,1e-5)',
    max_dets=15,
    peak_thres=0.001,
    ols_thres=0.7,           #ols_thres=0.3,peak_thres=0.4
    mnet_cfg=(4, 3),
    dcn=False,
)#0.5dic

confmap_cfg = dict(
    confmap_sigmas={
        'pedestrian': 20,              #'pedestrian': 15, 
        'cyclist': 20,
        'car': 30,
        # 'van': 40,
        # 'truck': 50,
    },
    confmap_sigmas_interval={
        'pedestrian': [5, 15],
        'cyclist': [10, 25],
        'car': [10, 30],                       #'cyclist': [8, 20],
        # 'van': [15, 40],
        # 'truck': [20, 50],
    },
    confmap_length={
        'pedestrian': 1,
        'cyclist': 2,
        'car': 3,
        # 'van': 4,
        # 'truck': 5,
    }
)

train_cfg = dict(
    n_epoch=50,
    batch_size=32,
    lr=1e-4,
    lr_step=10,  # lr will decrease 10 times after lr_step epoches
    win_size=4,
    train_step=1,
    train_stride=1,
    log_step=50,
    save_step=1000,
)
test_cfg = dict(
    test_step=1,
    test_stride=8,
    rr_min=1.0,  # min radar range
    rr_max=20.0,  # max radar range
    ra_min=-60.0,  # min radar angle
    ra_max=60.0,  # max radar angle
)
# python tools/prepare_dataset/prepare_data.py --config configs/config_rodnet_cdcv2_win16_mnet.py --data_root /media/zdk/SYQ/Radar_detection/RODNet-master --split train,test --out_data_dir /media/zdk/SYQ/Radar_detection/RODNet-master/data/pkl
# python tools/prepare_dataset/prepare_data.py --split train,test 
#CUDA_VISIBLE_DEVICES=1 python test_te
#tmux a -t 1