# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2

class SSIM(object):
    def __init__(self, k1=0.01, k2=0.02, L=1, window_size=11):
        self.k1 = k1
        self.k2 = k2           # constants for stable
        self.L = L             # the value range of input image pixels
        self.WS = window_size

    def _tf_fspecial_gauss(self, size, sigma=1.5):
        """Function to mimic the 'fspecial' gaussian MATLAB function"""
        x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.expand_dims(x_data, axis=-1)

        y_data = np.expand_dims(y_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)

        x = tf.constant(x_data, dtype=tf.float32)
        y = tf.constant(y_data, dtype=tf.float32)

        g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g / tf.reduce_sum(g)

    def ssim_loss(self, img1, img2):
        """
        The function is to calculate the ssim score
        """
        window = self._tf_fspecial_gauss(size=self.WS)  # output size is (window_size, window_size, 1, 1)
        #import pdb
        #pdb.set_trace()

        (_, _, _, channel) = img1.shape.as_list()

        window = tf.tile(window, [1, 1, channel, 1])

        # here we use tf.nn.depthwise_conv2d to imitate the group operation in torch.nn.conv2d
        mu1 = tf.nn.depthwise_conv2d(img1, window, strides = [1, 1, 1, 1], padding = 'VALID')
        mu2 = tf.nn.depthwise_conv2d(img2, window, strides = [1, 1, 1, 1], padding = 'VALID')

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        img1_2 = img1*img1#tf.pad(img1*img1, [[0,0], [0, self.WS//2], [0, self.WS//2], [0,0]], "CONSTANT")
        sigma1_sq = tf.subtract(tf.nn.depthwise_conv2d(img1_2, window, strides = [1 ,1, 1, 1], padding = 'VALID') , mu1_sq)
        img2_2 = img2*img2#tf.pad(img2*img2, [[0,0], [0, self.WS//2], [0, self.WS//2], [0,0]], "CONSTANT")
        sigma2_sq = tf.subtract(tf.nn.depthwise_conv2d(img2_2, window, strides = [1, 1, 1, 1], padding = 'VALID') ,mu2_sq)
        img12_2 = img1*img2#tf.pad(img1*img2, [[0,0], [0, self.WS//2], [0, self.WS//2], [0,0]], "CONSTANT")
        sigma1_2 = tf.subtract(tf.nn.depthwise_conv2d(img12_2, window, strides = [1, 1, 1, 1], padding = 'VALID') , mu1_mu2)

        c1 = (self.k1*self.L)**2
        c2 = (self.k2*self.L)**2

        ssim_map = ((2*mu1_mu2 + c1)*(2*sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1)*(sigma1_sq + sigma2_sq + c2))
        #print('ssim_map:', ssim_map)
        return tf.reduce_mean(ssim_map)

if __name__ == "__main__":

    #img1 = tf.ones((1,128,128,3))
    #img2 = tf.ones((1,128,128,3))
    DIR = '/home/siat-video/data/MCL-JCI/distorted_image'
    ref_img = DIR + '/ImageJND_SRC07/ImageJND_SRC07_100.jpg'
    dis_img = DIR + '/ImageJND_SRC07/ImageJND_SRC07_001.jpg'

    img1 = cv2.imread(ref_img).astype('float32')
    img2 = cv2.imread(dis_img).astype('float32')
    # results using tensorflow
    loss_ssim = SSIM(k1=0.01, k2=0.02, L=255, window_size=11)
    img1_tensor = tf.expand_dims(tf.constant(img1), 0)
    img2_tensor = tf.expand_dims(tf.constant(img2), 0)
    print('img1_tensor:', img1_tensor.shape)
    print('img2_tensor:', img2_tensor.shape)
    loss_results = loss_ssim.ssim_loss(img1_tensor, img2_tensor)
    sess = tf.Session()
    print(1-sess.run(loss_results))
