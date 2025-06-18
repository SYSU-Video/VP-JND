# -*- coding:utf-8 –*-

import tensorflow as tf
import os
import MCL_JCI_input, MCL_JCI
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

def tf_read_image(path):
    image_raw_data_jpg = tf.gfile.FastGFile(path, 'rb').read()
    img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)
    img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.uint8)

    return img_data_jpg


def random_sample(images_x, images_y, patch_size, num_patches):
    """random sample patch pairs from image pairs.
    从图像中随机选取块
    Args:
        :param images_x, images_y: tensor - (batch_size, height, width, depth).
        :param patch_size: int.
        :param num_patches: int. we crop num_patches patches from each image pair.

    Returns:
        :return patches_x, patches_y: tensor - (batch_size*num_patches, patch_size, patch_size, depth).
    """
    patches_x = []
    patches_y = []

    images_xy = tf.concat([images_x, images_y], axis=2)
    for j in range(num_patches):
        # Randomly crop a [height, width] section of the image.
        patch_xy = tf.random_crop(images_xy[:, :, :], [patch_size, patch_size, 3 * 2])

        patches_x.append(patch_xy[:, :, :3])
        patches_y.append(patch_xy[:, :, 3:])

    patches_x = tf.convert_to_tensor(value=patches_x, dtype=tf.float32, name='sampled_patches_x')
    patches_y = tf.convert_to_tensor(value=patches_y, dtype=tf.float32, name='sampled_patches_y')

    return patches_x, patches_y


def distort_input(comparison, anchor):
    # 输入：失真图像，参考图像
    comparison_img = tf_read_image(comparison)
    anchor_img = tf_read_image(anchor)

    distorted_comparison_img = tf.cast(comparison_img, tf.float32) * (1. / 255) - 0.5
    distorted_anchor_img = tf.cast(anchor_img, tf.float32) * (1. / 255) - 0.5

    comparison_patches, anchor_patches = random_sample(distorted_comparison_img, distorted_anchor_img,
                                                       patch_size=32,
                                                       num_patches=32)

    return comparison_patches, anchor_patches


def lossyOrlossless_predict(comparison1, anchor1):

    # predict  whether compressed image comparison1 is perceptually lossy from reference anchor1 or not
    # 输入参数：失真图像，参考图像
    # 输出：当失真图像失真人眼可见输出1，否则输出0
    graph = tf.Graph()
    comparison2 = str(comparison1)
    anchor2 = str(anchor1)
    comparison = comparison2[2:-2]
    anchor = anchor2[2:-2]
    #print(comparison)
    with graph.as_default() as g:
        keep_prob = tf.placeholder(tf.float32, name='ratio')
        patches_x, patches_y = distort_input(comparison2, anchor2)  # select patches from comparison and anchor

        # inference model.
        scores = MCL_JCI.inference(patches_x, patches_y, keep_prob)

        label_pre = tf.cast(tf.round(tf.nn.sigmoid(scores)), tf.int64)  # [1,0,1,1]<0.5 label: 0 and >0.5 label:1

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            MCL_JCI.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

    with tf.Session(graph=graph) as sess:
        # checkpoint_file = os.path.join(MCL_JCI_input.LOG_DIR, 'best_model.ckpt')
        # saver.restore(sess, checkpoint_file)
        #saver.restore(sess, './logs/best_model.ckpt')  # 路径
        saver.restore(sess, './logs/best_model.ckpt')  # 路径
        label = sess.run(label_pre, feed_dict={keep_prob: 1.0})
        return_value = int(label)
        #print(label)
        #print(return_value)
    return return_value
import os

import time
if __name__ == '__main__':
    #anchor_img = './QP9.jpg'
    #compar_img = './QP1.jpg'
    _img1 = '/home/siat-video/data/MCL-JCI/distorted_image/ImageJND_SRC01/ImageJND_SRC01_027.jpg'
    _img2 = '/home/siat-video/data/MCL-JCI/distorted_image/ImageJND_SRC01/ImageJND_SRC01_036.jpg'
    #print(_img1, _img2)
    start_time = time.time()
    pre_label = lossyOrlossless_predict(_img1, _img2)
    print("--- %s seconds ---" % (time.time() - start_time))
    #print(pre_label)
    print("Image JND lossy_or_lossless:", pre_label)
