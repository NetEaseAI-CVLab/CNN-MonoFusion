import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import sys
sys.path.append("..")

from decimal import Decimal
import time
from tqdm import tqdm

# gpu config
# os.environ["CUDA_VISIBLE_DEVICES"]='1'

depth_factor = 13107
depth_factor_inv = 1.0 / depth_factor
nyu_depth_factor_inv = 0.001
max_depth = 10
depth_gradient_thr = 0.09
img_gradient_thr = 10
nyu_focal = 1.49333

IMG_MEAN = np.array(
    (104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_path', type=str, help='Converted parameters for the model')
parser.add_argument(
    '--image_dir', type=str, help='Directory of images to predict')
parser.add_argument(
    '--save_dir', type=str, help='Directory of prediction depth to save')
parser.add_argument('--pred_depth_dir_name', type=str, help='depth dir name')
parser.add_argument('--figure_dir_name', type=str, help='figure save dir')
parser.add_argument('--rgb_extend_name', type=str, help='rgb extend name')
parser.add_argument('--depth_extend_name', type=str, help='depth extend name')
parser.add_argument('--zfill_length', type=int, help='name fill zero nums')
parser.add_argument('--model_type', type=str, help='Choose Model Type')
parser.add_argument('--net_height', type=int, help='network height')
parser.add_argument('--net_width', type=int, help='network width')
parser.add_argument('--image_height', type=int, help='image height')
parser.add_argument('--image_width', type=int, help='image width')
parser.add_argument(
    '--eval_dataset_type', type=str, help='evaluate dataset type')
parser.add_argument(
    '--is_evaluate',
    help='if set, save depths.npy for evaluate',
    action='store_true',
    default=False)
args = parser.parse_args()


def progress_display(current, total, external_str):
    progress = Decimal(((current + 1) / float(total)) * 100.0).quantize(
        Decimal('0.00'))
    stdout_msg = external_str + str(progress) + '%' + "\r"
    sys.stdout.write(stdout_msg)
    sys.stdout.flush()


def gradient(img_gray):
    gx = np.gradient(img_gray, axis=0)
    gy = np.gradient(img_gray, axis=1)
    g = gx * gx + gy * gy
    return np.sqrt(g)


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


def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def predict_process(sess, net, input_node, image_dir, width, height):
    sequence_list = os.listdir(image_dir)
    sequence_list.sort()
    for sequence in sequence_list:
        output_dir = args.save_dir + sequence + args.pred_depth_dir_name + '/'
        output_figure_dir = args.save_dir + sequence + args.figure_dir_name + '/'
        eva_npy_save_dir = args.save_dir + 'depths_npy/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(output_figure_dir):
            os.makedirs(output_figure_dir)
        if not os.path.exists(eva_npy_save_dir):
            os.makedirs(eva_npy_save_dir)

        # Predict the image
        sequence_dir = args.image_dir + sequence + "/"
        pred_depth_npy_path = eva_npy_save_dir + sequence + '_pred_depths.npy'
        gt_depth_npy_path = eva_npy_save_dir + sequence + '_gts_depths.npy'

        # list all files
        files = os.listdir(sequence_dir)
        files.sort()
        rgb_name_list = []
        depth_name_list = []

        print "\n------------------------------------------------------------------------------------------------------------"
        print "Waiting for loading new sequence ......"
        for file_name in tqdm(files):
            if file_name.split('_')[1] == args.rgb_extend_name.split('_')[1]:
                rgb_name_list.append(file_name)
            if file_name.split('_')[1] == args.depth_extend_name.split('_')[1]:
                depth_name_list.append(file_name)

        image_nums = len(
            rgb_name_list)  # min(len(rgb_name_list), len(depth_name_list))
        print "Loading done!"
        print "- Sequence Dir: ", sequence_dir
        print "- Image Nums: ", image_nums
        print "- Save Dir: ", output_dir
        print "- Input Node: ", input_node
        print "Start predicting ......"

        if image_nums < 40:
            print "Too Small Image Nums! Skip This Sequence."
            continue

        pred_depths = np.zeros((image_nums, 270, 480), dtype=np.float32)
        gt_depths = np.zeros((image_nums, 270, 480), dtype=np.float32)
        for i in range(image_nums):
            frame_id = str(i).zfill(args.zfill_length)
            image_path = sequence_dir + frame_id + args.rgb_extend_name

            preprocess_start_time = time.time()
            img = cv2.imread(image_path)
            img_resize = cv2.resize(img, (width, height))
            img_resize = img_resize - IMG_MEAN
            img_resize_expend = np.expand_dims(np.asarray(img_resize), axis=0)

            # Evalute the network for the given image
            forward_start_time = time.time()
            pred = sess.run(
                net.get_output(), feed_dict={
                    input_node: img_resize_expend
                })
            forward_end_time = time.time()
            forward_duration = forward_end_time - forward_start_time
            im_preprocess_duration = forward_start_time - preprocess_start_time

            progress_external_str = 'Predicting image ' + frame_id + args.rgb_extend_name + ', img pre-process cost {:.6f}sec ' + ', forward cost {:.6f}sec' + ' ,  progress:  '
            progress_display(i, image_nums,
                             progress_external_str.format(
                                 im_preprocess_duration, forward_duration))

            # depth save path
            depth_pred_save_path = output_dir + frame_id + "_adenet_depth_pred.png"
            img_resize_save_path = output_dir + frame_id + "_color_resize.png"

            # scale: 65535/max_depth
            # filter
            depth_pred_origin = pred[0, :, :, 0]
            depth_pred = np.copy(depth_pred_origin)
            depth_pred[depth_pred > max_depth] = 0
            
            out_height, out_width = depth_pred_origin.shape
            image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image_gray_resize = cv2.resize(image_gray, (out_width, out_height))
            image_arr_gray = np.array(image_gray_resize).astype(np.float32)
            image_arr_gradient = gradient(image_arr_gray)

            # resize depth
            depth_pred_resize = cv2.resize(
                depth_pred, (480, 270), interpolation=cv2.INTER_NEAREST)
            depth_pred_resize_float = depth_pred_resize
            depth_pred_resize = depth_pred_resize * depth_factor
            depth_pred_resize_around = np.around(depth_pred_resize)
            depth_pred_resize_around = depth_pred_resize_around.astype(np.uint16)
            depth_pred_resize_around[:, :70] = 0
            depth_pred_resize_around[:, 410:] = 0
            # cv2.imwrite(depth_pred_save_path, depth_pred_resize_around)

            # resize img
            cv_img = cv2.imread(image_path)
            cv_img_resize = cv2.resize(cv_img, (480, 270), interpolation=cv2.INTER_LINEAR)

            if args.is_evaluate == True:
                pred_depths[i] = np.copy(depth_pred_resize_float)
                depth_path = sequence_dir + frame_id + args.depth_extend_name
                gt_depth = cv2.imread(depth_path, cv2.CV_LOAD_IMAGE_UNCHANGED)
                gt_depth_float = np.copy(gt_depth * depth_factor_inv)
                gt_depths[i] = gt_depth_float

        if args.is_evaluate == True:
            print "Waiting for depths.npy save ......"
            np.save(pred_depth_npy_path, pred_depths)
            np.save(gt_depth_npy_path, gt_depths)
            print "Save npy done!"
        print "Sequence ", sequence, " predict done!\n"


def predict_nyu_process(sess, net, input_node, image_dir, width, height):
    sequence_list = os.listdir(image_dir)
    sequence_list.sort()
    for sequence in sequence_list:
        output_dir = args.save_dir + sequence + args.pred_depth_dir_name + '/'
        output_figure_dir = args.save_dir + sequence + args.figure_dir_name + '/'
        eva_npy_save_dir = args.save_dir + 'depths_npy/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(output_figure_dir):
            os.makedirs(output_figure_dir)
        if not os.path.exists(eva_npy_save_dir):
            os.makedirs(eva_npy_save_dir)

        # Predict the image
        sequence_dir = args.image_dir + sequence + "/"
        pred_depth_npy_path = eva_npy_save_dir + sequence + '_pred_depths.npy'
        gt_depth_npy_path = eva_npy_save_dir + sequence + '_gts_depths.npy'

        # list all files
        files = os.listdir(sequence_dir)
        files.sort()
        rgb_name_list = []
        depth_name_list = []

        print "\n------------------------------------------------------------------------------------------------------------"
        print "Waiting for loading new sequence ......"
        for file_name in tqdm(files):
            if file_name.split('_')[1] == args.rgb_extend_name.split('_')[1]:
                rgb_name_list.append(file_name)
            if file_name.split('_')[1] == args.depth_extend_name.split('_')[1]:
                depth_name_list.append(file_name)

        image_nums = len(rgb_name_list)  # min(len(rgb_name_list), len(depth_name_list))
        print "Loading done!"
        print "- Sequence Dir: ", sequence_dir
        print "- Image Nums: ", image_nums
        print "- Save Dir: ", output_dir
        print "- Input Node: ", input_node
        print "Start predicting ......"
        
        images = os.listdir(sequence_dir)
        images.sort()
        image_max_number = int(images[-1].split('_')[0])
        print 'images max number in image_name: ', image_max_number
        count_lost = 0
        pred_depths = np.zeros(
            (image_nums, args.image_height, args.image_width), dtype=np.float32)
        gt_depths = np.zeros(
            (image_nums, args.image_height, args.image_width), dtype=np.float32)
        for i in range(image_max_number):
            frame_id = str(i).zfill(args.zfill_length)
            image_path = sequence_dir + frame_id + args.rgb_extend_name

            if os.path.exists(image_path):
                preprocess_start_time = time.time()
                
                # adenet author use rgb
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                img_resize = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

                img_resize = img_resize - IMG_MEAN

                img_resize_expend = np.expand_dims(np.asarray(img_resize), axis=0)

                # Evalute the network for the given image
                forward_start_time = time.time()
                pred = sess.run(net.get_output(), feed_dict={input_node: img_resize_expend})
                forward_end_time = time.time()
                forward_duration = forward_end_time - forward_start_time
                im_preprocess_duration = forward_start_time - preprocess_start_time

                progress_external_str = 'Predicting image ' + frame_id + args.rgb_extend_name + ', img pre-process cost {:.6f}sec ' + ', forward cost {:.6f}sec' + ' ,  progress:  '
                progress_display(i, image_max_number,
                                 progress_external_str.format(
                                     im_preprocess_duration, forward_duration))

                # depth save path
                depth_pred_save_path = output_dir + frame_id + "_adenet_depth_pred.png"
                img_resize_save_path = output_dir + frame_id + "_color_resize.png"

                # scale: 65535/max_depth
                # filter
                depth_pred_origin = pred[0, :, :, 0]
                depth_pred = np.copy(depth_pred_origin)
                depth_pred = depth_pred * nyu_focal
                depth_pred[depth_pred > max_depth] = 0
                # depth_pred_gradient = gradient(depth_pred_origin)
                # depth_pred[depth_pred_gradient>depth_gradient_thr] = 0

                out_height, out_width = depth_pred_origin.shape

                # resize depth
                depth_pred_resize = cv2.resize(
                    depth_pred, (args.image_width, args.image_height),
                    interpolation=cv2.INTER_NEAREST)
                depth_pred_resize_float = depth_pred_resize
                depth_pred_resize = depth_pred_resize * depth_factor
                depth_pred_resize_around = np.around(depth_pred_resize)
                depth_pred_resize_around = depth_pred_resize_around.astype(np.uint16)

                # resize img
                cv_img = cv2.imread(image_path)
                cv_img_resize = cv2.resize(cv_img, (args.image_width, args.image_height), interpolation=cv2.INTER_LINEAR)
                # cv2.imwrite(img_resize_save_path, cv_img_resize)

                if args.is_evaluate == True:
                    pred_depths[i - count_lost] = np.copy(depth_pred_resize_float)
                    depth_path = sequence_dir + frame_id + args.depth_extend_name
                    gt_depth = cv2.imread(depth_path, cv2.CV_LOAD_IMAGE_UNCHANGED)
                    gt_depth_resize = cv2.resize(gt_depth, (args.image_width, args.image_height), interpolation=cv2.INTER_NEAREST)
                    gt_depth_float = np.copy(gt_depth_resize * nyu_depth_factor_inv)
                    gt_depth_float[gt_depth_float > max_depth] = 0
                    gt_depths[i - count_lost] = gt_depth_float
            else:
                count_lost = count_lost + 1
                continue
        if args.is_evaluate == True:
            print "Waiting for depths.npy save ......"
            np.save(pred_depth_npy_path, pred_depths)
            np.save(gt_depth_npy_path, gt_depths)
            print "Save npy done!"
        print "Sequence ", sequence, " predict done!\n"


def predict_astrous_os8_concat(model_data_path, image_dir, eval_dataset_type):
    import adenet_def

    INPUT_SIZE = '160,240'
    height = 160
    width = 240

    # Default input size
    channels = 3
    batch_size = 1

    # Create a placeholder for the input images
    input_node = tf.placeholder(
        tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = adenet_def.ResNet50_astrous_concat(
        {
            'data': input_node
        }, batch_size, 1, False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Load the converted parameters
        print('Loading the astrous model')

        loader = tf.train.Saver()
        load(loader, sess, model_data_path)

        print "load done!"

        print "\n------------------------------------------------------------------------------------------------------------"
        print "- Model Path: ", args.model_path
        print "- depth_extend_name: ", args.depth_extend_name
        print "- zfill length: ", args.zfill_length
        print "- Is save depths.npy for evaluate: ", args.is_evaluate
        print "------------------------------------------------------------------------------------------------------------\n"

        if eval_dataset_type == "NYU_VAL":
            predict_nyu_process(sess, net, input_node, image_dir, width, height)
        else:
            predict_process(sess, net, input_node, image_dir, width, height)
        print "\nAll sequence predict done!\n"


def main():
    if os.path.exists(args.image_dir):
        if args.model_type == "adenet_astrous_os8_concat":
            predict_astrous_os8_concat(args.model_path, args.image_dir,
                                       args.eval_dataset_type)
    os._exit(0)


if __name__ == '__main__':
    main()
