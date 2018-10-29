import numpy as np
from evaluate_util import *
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Evaluation on the KITTI dataset')
parser.add_argument(
    '--depths_npy_dir',
    type=str,
    help='path to pred depths np file',
    required=True)

args = parser.parse_args()

# pred_np_path = args.depths_npy_dir+'pred_depths.npy'
# gt_np_path = args.depths_npy_dir+'gts_depths.npy'
# print '--------------------------------------------------------------------------'
# print '- Pred Depth Numpy-File Path: ', pred_np_path
# print '- GT Depth Numpy-File Path: ', gt_np_path
# print '--------------------------------------------------------------------------'


def depth_figure_save(dict):
    fig = plt.figure(figsize=(25, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(dict['src_img1'])
    ax1.axis('off')
    ax1.set_title(dict['img1_name'])

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(dict['src_img2'])
    ax2.axis('off')
    ax2.set_title(dict['img2_name'])

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(dict['src_img3'])
    ax3.axis('off')
    ax3.set_title(dict['img3_name'])

    ax3 = fig.add_subplot(2, 2, 4)
    ax3.imshow(dict['src_img4'])
    ax3.axis('off')
    ax3.set_title(dict['img4_name'])

    #print("figure save in : " + save_path + "\n")
    fig.savefig(dict['save_path'])
    plt.close(fig)


# refer to [monodepth](https://github.com/mrharicot/monodepth/blob/master/utils/evaluate_kitti.py)
if __name__ == '__main__':

    npy_files = os.listdir(args.depths_npy_dir)
    npy_files.sort()
    dataset_nums = len(npy_files) / 2
    rms_total = np.zeros(dataset_nums, np.float32)
    log_rms_total = np.zeros(dataset_nums, np.float32)
    abs_rel_total = np.zeros(dataset_nums, np.float32)
    sq_rel_total = np.zeros(dataset_nums, np.float32)
    d1_all_total = np.zeros(dataset_nums, np.float32)
    a1_total = np.zeros(dataset_nums, np.float32)
    a2_total = np.zeros(dataset_nums, np.float32)
    a3_total = np.zeros(dataset_nums, np.float32)
    log10_total = np.zeros(dataset_nums, np.float32)

    print '------------------------------------------------------------------------------------'
    for npy_idx in range(len(npy_files)):

        npy_file = npy_files[npy_idx]
        # print npy_file
        if '_pred_depths' in npy_file:
            sequence_name = npy_file.split('_')[0]
            pred_np_path = args.depths_npy_dir + sequence_name + '_pred_depths.npy'
            gt_np_path = args.depths_npy_dir + sequence_name + '_gts_depths.npy'

            # load processed pred & gt depths.np generate by predict_sequence.py
            pred_depths = np.load(pred_np_path)
            gt_depths = np.load(gt_np_path)
            num_samples = pred_depths.shape[0]

            print '- Depth np file idx: ', npy_idx, 'Sequence name: ', sequence_name, ', Evaluate image nums: ', num_samples

            rms = np.zeros(num_samples, np.float32)
            log_rms = np.zeros(num_samples, np.float32)
            abs_rel = np.zeros(num_samples, np.float32)
            sq_rel = np.zeros(num_samples, np.float32)
            d1_all = np.zeros(num_samples, np.float32)
            a1 = np.zeros(num_samples, np.float32)
            a2 = np.zeros(num_samples, np.float32)
            a3 = np.zeros(num_samples, np.float32)
            log10_error = np.zeros(num_samples, np.float32)
            
            for i in range(0, num_samples - 1):
                gt_depth = gt_depths[i]
                pred_depth = pred_depths[i]
                gt_depth_copy = np.copy(gt_depth)
                pred_depth_copy = np.copy(pred_depth)
                gt_depth_copy[pred_depth < 0.03] = 0
                pred_depth_copy[gt_depth < 0.03] = 0

                mask = gt_depth_copy > 0.03
                disp_diff = np.abs(gt_depth_copy[mask] - pred_depth_copy[mask])
                bad_pixels = np.logical_and(
                    disp_diff >= 0.04,
                    (disp_diff / gt_depth_copy[mask]) >= 0.05)

                d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

                abs_rel[i], sq_rel[i], rms[i], log_rms[
                    i], a1[i], a2[i], a3[i], log10_error[i] = compute_errors(
                        gt_depth_copy[mask], pred_depth_copy[mask])

            print(
                "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".
                format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1',
                       'a2', 'a3','log10'))
            print(
                "{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".
                format(abs_rel.mean(), sq_rel.mean(), rms.mean(),
                       log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(),
                       a3.mean(), log10_error.mean()))
            rms_total[npy_idx / 2] = rms.mean()
            log_rms_total[npy_idx / 2] = log_rms.mean()
            abs_rel_total[npy_idx / 2] = abs_rel.mean()
            sq_rel_total[npy_idx / 2] = sq_rel.mean()
            d1_all_total[npy_idx / 2] = d1_all.mean()
            a1_total[npy_idx / 2] = a1.mean()
            a2_total[npy_idx / 2] = a2.mean()
            a3_total[npy_idx / 2] = a3.mean()
            log10_total[npy_idx / 2] = log10_error.mean()
            

    print '\n------------------------------------------------------------------------------------'
    print 'All Depths Npy Mean: '
    print(
        "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(
            'abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3', 'log10'))
    print(
        "{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".
        format(abs_rel_total.mean(), sq_rel_total.mean(), rms_total.mean(),
               log_rms_total.mean(), d1_all_total.mean(), a1_total.mean(),
               a2_total.mean(), a3_total.mean(), log10_total.mean()))
    print 'Evaluate All Depth.npy Done!'
