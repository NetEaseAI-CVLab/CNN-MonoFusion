# Copyright @2018 The CNN_MonoFusion Authors (NetEaseAI-CVlab). 
# All Rights Reserved.
#
# Please cited our paper if you find CNN_MonoFusion useful in your research!
#
# See the License for the specific language governing permissions
# and limitations under the License.
#

from __future__ import print_function
import numpy as np
import tensorflow as tf
import socket
import cv2
import time

import sys
sys.path.append("..")

import os
import argparse
import adenet_def
# socket parameters
address = ('', 6666)

# gpu config
# os.environ["CUDA_VISIBLE_DEVICES"]='1'

# depth parameters
depth_factor = 13107
depth_factor_inv = 1.0/depth_factor
max_depth = 4.5
depth_gradient_thr = 0.2
img_gradient_thr = 10
INPUT_SIZE = '160,240'
height = 160
width = 240
channels = 3
batch_size = 1
black_hole_width = 25

# camera parameters
cx_gt = 492.247
cy_gt = 263.355
focal_scale = 1.0
nyu_focal = 1.49333
# img mean
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

model_path = '../models/adenet_merge_nyu_kinect_tum/neair-adenet-final'


def gradient(img_gray):
    gx = np.gradient(img_gray, axis=0)
    gy = np.gradient(img_gray, axis=1)
    g = gx*gx + gy*gy
    return np.sqrt(g)

    
def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

    
def send_depth(conn, depth):
    result, encoded_depth = cv2.imencode('.png', depth)
    str_depth = encoded_depth.tostring()
    str_length = str(len(str_depth)).ljust(16).encode()
    conn.sendall(str_length)
    conn.sendall(str_depth)


def receive_image(conn):
    str_length = receive_all(conn, 16)
    if str_length is None:
        return None
    str_length = str_length.decode('utf-8')
    length = int(str_length)
    # print 'recv length', length
    buf = receive_all(conn, length)
    if buf is None:
        return None
    encoded_img = np.fromstring(buf, dtype=np.uint8)
    img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    return img


def receive_all(conn, count):
    buf = b''
    while count:
        new_buf = conn.recv(count)
        if not new_buf:
            return None
        buf += new_buf
        count -= len(new_buf)
    return buf
    
 
def main():
    print("Adaptive-depth-esti-net server socket !")

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))
    
    # Construct the network
    net = adenet_def.ResNet50_astrous_concat({'data': input_node}, batch_size, 1, False)


    # SESSION
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    
    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    # SAVER 
    print('Loading astrous-os8-concat model ......')
    loader = tf.train.Saver()
    load(loader, sess, model_path)
    print('Load done!')

    # SOCKET
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(address)
    while True:
        print('Wait for connection...')
        s.listen(True)
        conn, addr = s.accept()
        print('Accept a connection.')

        # receive image size of client
        depth_thr=0.1
        str_depth_thr = receive_all(conn, 16)
        str_focal_scale = receive_all(conn, 16)
        if str_depth_thr is None:
            print('Can not receive client\'s image parameters. Use ground truth instead.')
            depth_thr = 0.1
        else:
            str_depth_thr = str_depth_thr.decode('utf-8')
            depth_thr = float(str_depth_thr)
            print('User Config Depth_Gradient_Thr: ', depth_thr)
        
        focal_scale = 1.0
        if str_focal_scale is None:
            print('Can not receive client\'s image parameters of focal_scale. Use default instead.')
            focal_scale = 1.0
        else:
            str_focal_scale = str_focal_scale.decode('utf-8')
            focal_scale = float(str_focal_scale)
            print('User Config focal_scale: ', focal_scale)

        # img_id = 0
        try:
            while True:
                img = receive_image(conn)
                
                if img is None:
                    print('Connection is closed.')
                    break
                
                # img pre-process
                preprocess_start_time = time.time()
                img_height = img.shape[0]
                img_width = img.shape[1]
                
                img_resize = img - IMG_MEAN
                img_resize_expend = np.expand_dims(np.asarray(img_resize), axis = 0)
                
                # adenet predict
                forward_start_time = time.time()
                pred = sess.run(net.get_output(), feed_dict={input_node: img_resize_expend})
                forward_end_time = time.time()
                
                # pred-depth process 
                depth_pred_origin = pred[0,:,:,0] 
                depth_pred = np.copy(depth_pred_origin)
                depth_pred[depth_pred>max_depth] = 0
                
                depth_pred = depth_pred * focal_scale
                depth_pred_gradient = gradient(depth_pred_origin)  
                depth_pred[depth_pred_gradient>depth_thr] = 0

                
                # depth around
                depth_pred_scale = depth_pred*depth_factor
                depth_pred_around = np.around(depth_pred_scale)
                depth_pred_around = depth_pred_around.astype(np.uint16)
                # depth_pred_resize = cv2.resize(depth_pred_around, (480,270), interpolation=cv2.INTER_NEAREST)
                # depth_pred_resize[:, :70] = 0
                # depth_pred_resize[:, (480-70):] = 0
                depth_process_end_time = time.time()
                
                # cost time
                forward_duration = forward_end_time - forward_start_time
                im_preprocess_duration = forward_start_time - preprocess_start_time
                depth_process_duration = depth_process_end_time - forward_end_time
                
                # print time
                print('recv img size: ', img.shape, ', depth-pred size: ', depth_pred_around.shape)
                cost_time_str = 'img preprocess {:.6f}sec ' + ', forward {:.6f}sec' + ', depth process {:.6f}sec '
                sys_out_msg = cost_time_str.format(im_preprocess_duration, forward_duration, depth_process_duration)
                # sys.stdout.write(sys_out_msg)
                # sys.stdout.flush()
                print(sys_out_msg)
                
                # send depth
                # print 'depth size: ', depth_pred_resize.shape
                send_depth(conn,depth_pred_around)
                
                
        except:
            pass

            
if __name__ == '__main__':
    main()
