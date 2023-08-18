import numpy as np
import matplotlib.pyplot as plt
from rodnet.core.post_processing.lnms import get_ols_btw_objects
import argparse

from cruw import CRUW

from rodnet.datasets.CRDataset import CRDataset
from rodnet.datasets.collate_functions import cr_collate
from rodnet.core.post_processing import post_process, post_process_single_frame
from rodnet.core.post_processing import write_dets_results, write_dets_results_single_frame
from rodnet.core.post_processing import ConfmapStack
from rodnet.core.radar_processing import chirp_amp
from rodnet.utils.visualization import visualize_test_img, visualize_test_img_wo_gt
from rodnet.utils.load_configs import load_configs_from_file, parse_cfgs, update_config_dict
from rodnet.utils.solve_dir import create_random_model_name
from rodnet.models.M_UHRNet import RODNetUNet as RODNet
def post_process(confmaps, config_dict):
    """
    Post-processing for RODNet
    :param confmaps: predicted confidence map [B, n_class, win_size, ramap_r, ramap_a]
    :param search_size: search other detections within this window (resolution of our system)
    :param peak_thres: peak threshold
    :return: [B, win_size, max_dets, 4]
    """
    n_class = 3
    model_configs = config_dict['model_cfg']
    rng_grid = config_dict['mappings']['range_grid']
    agl_grid = config_dict['mappings']['angle_grid']
    max_dets = model_configs['max_dets']
    peak_thres = model_configs['peak_thres']

    batch_size, class_size, win_size, height, width = confmaps.shape

    if class_size != n_class:
        raise TypeError("Wrong class number setting. ")

    res_final = - np.ones((batch_size, win_size, max_dets, 4))

    for b in range(batch_size):
        for w in range(win_size):
            detect_mat = []
            for c in range(class_size):
                obj_dicts_in_class = []
                confmap = np.squeeze(confmaps[b, c, w, :, :])
                rowids, colids = detect_peaks(confmap, threshold=peak_thres)

                for ridx, aidx in zip(rowids, colids):
                    rng = rng_grid[ridx]
                    agl = agl_grid[aidx]
                    conf = confmap[ridx, aidx]
                    obj_dict = {'frameid': None, 'range': rng, 'angle': agl, 'ridx': ridx, 'aidx': aidx,
                                'classid': c, 'score': conf}
                    obj_dicts_in_class.append(obj_dict)

                detect_mat_in_class = lnms(obj_dicts_in_class, config_dict)
                detect_mat.append(detect_mat_in_class)

            detect_mat = np.array(detect_mat)
            detect_mat = np.reshape(detect_mat, (class_size * max_dets, 4))
            detect_mat = detect_mat[detect_mat[:, 3].argsort(kind='mergesort')[::-1]]
            res_final[b, w, :, :] = detect_mat[:max_dets]

    return res_final


def lnms(obj_dicts_in_class, dataset, config_dict):
    """
    Location-based NMS
    :param obj_dicts_in_class:
    :param config_dict:
    :return:
    """
    model_configs = config_dict['model_cfg']

    detect_mat = - np.ones((model_configs['max_dets'], 4))
    cur_det_id = 0
    # sort peaks by confidence score
    inds = np.argsort([-d['score'] for d in obj_dicts_in_class], kind='mergesort')
    dts = [obj_dicts_in_class[i] for i in inds]
    while len(dts) != 0:
        if cur_det_id >= model_configs['max_dets']:
            break
        p_star = dts[0]
        detect_mat[cur_det_id, 0] = p_star['class_id']
        detect_mat[cur_det_id, 1] = p_star['range_id']
        detect_mat[cur_det_id, 2] = p_star['angle_id']
        detect_mat[cur_det_id, 3] = p_star['score']
        cur_det_id += 1
        del dts[0]
        for pid, pi in enumerate(dts):
            ols = get_ols_btw_objects(p_star, pi, dataset)
            if ols > model_configs['ols_thres']:
                del dts[pid]

    return detect_mat



def detect_peaks(image, threshold=0.3):
    peaks_row = []
    peaks_col = []
    height, width = image.shape
    for h in range(1, height - 1):
        for w in range(2, width - 2):
            area = image[h - 1:h + 2, w - 2:w + 3]
            center = image[h, w]
            flag = np.where(area >= center)
            if flag[0].shape[0] == 1 and center > threshold:
                peaks_row.append(h)
                peaks_col.append(w)

    return peaks_row, peaks_col
def post_process_single_frame(confmaps, dataset, config_dict):
    """
    Post-processing for RODNet
    :param confmaps: predicted confidence map [B, n_class, win_size, ramap_r, ramap_a]
    :param search_size: search other detections within this window (resolution of our system)
    :param peak_thres: peak threshold
    :return: [B, win_size, max_dets, 4]
    """
    n_class = dataset.object_cfg.n_class
    rng_grid = dataset.range_grid
    agl_grid = dataset.angle_grid
    model_configs = config_dict['model_cfg']
    max_dets = model_configs['max_dets']
    peak_thres = model_configs['peak_thres']

    class_size, height, width = confmaps.shape

    if class_size != n_class:
        raise TypeError("Wrong class number setting. ")

    res_final = - np.ones((max_dets, 4))

    detect_mat = []
    for c in range(class_size):
        obj_dicts_in_class = []
        confmap = confmaps[c, :, :]
        rowids, colids = detect_peaks(confmap, threshold=peak_thres)

        for ridx, aidx in zip(rowids, colids):
            rng = rng_grid[ridx]
            agl = agl_grid[aidx]
            conf = confmap[ridx, aidx]
            obj_dict = dict(
                frame_id=None,
                range=rng,
                angle=agl,
                range_id=ridx,
                angle_id=aidx,
                class_id=c,
                score=conf,
            )
            obj_dicts_in_class.append(obj_dict)

        detect_mat_in_class = lnms(obj_dicts_in_class, dataset, config_dict)
        detect_mat.append(detect_mat_in_class)

    detect_mat = np.array(detect_mat)
    detect_mat = np.reshape(detect_mat, (class_size * max_dets, 4))
    detect_mat = detect_mat[detect_mat[:, 3].argsort(kind='mergesort')[::-1]]
    res_final[:, :] = detect_mat[:max_dets]

    return res_final
def visualize_postprocessing(confmaps, det_results):
    confmap_pred = np.transpose(confmaps, (1, 2, 0))
    plt.imshow(confmap_pred, vmin=0, vmax=1, origin='lower', aspect='auto')
    plt.show()
    for d in range(rodnet_configs['max_dets']):
        cla_id = int(det_results[d, 0])
        if cla_id == -1:
            continue
        row_id = det_results[d, 1]
        col_id = det_results[d, 2]
        conf = det_results[d, 3]
        cla_str = class_table[cla_id]
        plt.scatter(col_id, row_id, s=50, c='white')
        plt.text(col_id + 5, row_id, cla_str + '\n%.2f' % conf, color='white', fontsize=10, fontweight='black')
    plt.axis('off')
    plt.title("RODNet Detection")
    plt.show()
def parse_args():
    parser = argparse.ArgumentParser(description='Test RODNet.')

    parser.add_argument('--config', type=str, help='choose rodnet model configurations')
    parser.add_argument('--sensor_config', type=str, default='sensor_config_rod2021')
    parser.add_argument('--data_dir', type=str, default='/media/zdk/SYQ/Radar_detection/RODNet-master/data/pkl', help='directory to the prepared data')
    parser.add_argument('--checkpoint', type=str, help='path to the saved trained model')
    parser.add_argument('--res_dir', type=str, default='/media/zdk/SYQ/Radar_detection/RODNet-master/results', help='directory to save testing results')
    parser.add_argument('--use_noise_channel', action="store_true", help="use noise channel or not")
    parser.add_argument('--demo', action="store_true", help='False: test with GT, True: demo without GT')
    parser.add_argument('--symbol', action="store_true", help='use symbol or text+score')

    parser = parse_cfgs(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    sybl = args.symbol

    config_dict = load_configs_from_file('/media/zdk/SYQ/Radar_detection/RODNet-master/configs/config_rodnet_cdcv2_win16_mnet.py')
    config_dict = update_config_dict(config_dict, args)  # update configs by args

    dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'], sensor_config_name=args.sensor_config)
    radar_configs = dataset.sensor_cfg.radar_cfg
    range_grid = dataset.range_grid
    angle_grid = dataset.angle_grid

    model_cfg = config_dict['model_cfg']   
    
    
    input_test = np.random.random_sample((3, 128, 128))
    # a = input_test[0, :, 0, :, :]
    # b = input_test[1, :, 1, :, :]
    # print(input_test[0, :, 0, :, :])
    # print(input_test[1, :, 1, :, :])
    rodnet_configs={'max_dets': 20}
    rodnet_configs['max_dets'] = 20
    class_table = ['pedestrian','cyclist','car']
    res_final = post_process_single_frame(input_test, dataset, config_dict)
    print(res_final.size)
    print(res_final.shape)
    visualize_postprocessing(input_test, res_final)
    plt.show()
    # for b in range(1):
    #     for w in range(16):
    #         confmaps = np.squeeze(input_test[b, :, w, :, :])
    #         visualize_postprocessing(confmaps, res_final[b, w, :, :])
