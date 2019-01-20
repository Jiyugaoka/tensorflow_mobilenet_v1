#!/usr/bin/env python
# -*- coding:utf-8 -*-
#auther:jf183
#datetime:2019/1/2 21:12

import tensorflow as tf
import tensorflow.contrib as tc
import get_tensor_from_checkpoint as get_tensor

import numpy as np
import cv2
import time

class MobileNetV1(object):
    def __init__(self, input_size=224, classnum=6):
        self.input_size = input_size
        self.classnum = classnum
        self.normalizer = tc.layers.batch_norm

        with tf.variable_scope('MobilenetV1'):
            self._create_placeholders()
            self._build_model()


    def _create_placeholders(self):
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size, self.input_size, 3], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.classnum], name="input_y")
        self.is_training = tf.placeholder(tf.bool)
        self.bn_params = {'is_training': self.is_training, 'scope': 'BatchNorm', 'scale': True}

    def _build_model(self):
        i = 0
        self.conv1 = tc.layers.conv2d(self.input, num_outputs=32, kernel_size=3, stride=2,
                                    normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}'.format(i))

        # 1

        i += 1
        self.dconv1 = tc.layers.separable_conv2d(self.conv1, num_outputs=None, kernel_size=3, depth_multiplier=1,
                       normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_depthwise'.format(i))
        self.pconv1 = tc.layers.conv2d(self.dconv1, 64, 1, normalizer_fn=self.normalizer,
                       normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        # 2

        i += 1
        self.dconv2 = tc.layers.separable_conv2d(self.pconv1, None, 3, 1, 2,
                       normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_depthwise'.format(i))
        self.pconv2 = tc.layers.conv2d(self.dconv2, 128, 1, normalizer_fn=self.normalizer,
                       normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        # 3

        i += 1
        self.dconv3 = tc.layers.separable_conv2d(self.pconv2, None, 3, 1, 1,
                        normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_depthwise'.format(i))
        self.pconv3 = tc.layers.conv2d(self.dconv3, 128, 1, normalizer_fn=self.normalizer,
                        normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        # 4

        i += 1
        self.dconv4 = tc.layers.separable_conv2d(self.pconv3, None, 3, 1, 2,
                        normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_depthwise'.format(i))
        self.pconv4 = tc.layers.conv2d(self.dconv4, 256, 1, normalizer_fn=self.normalizer,
                        normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        # 5

        i += 1
        self.dconv5 = tc.layers.separable_conv2d(self.pconv4, None, 3, 1, 1,
                        normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_depthwise'.format(i))
        self.pconv5 = tc.layers.conv2d(self.dconv5, 256, 1, normalizer_fn=self.normalizer,
                        normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        # 6

        i += 1
        self.dconv6 = tc.layers.separable_conv2d(self.pconv5, None, 3, 1, 2,
                        normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_depthwise'.format(i))
        self.pconv6 = tc.layers.conv2d(self.dconv6, 512, 1, normalizer_fn=self.normalizer,
                        normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        # 7_1

        i += 1
        self.dconv71 = tc.layers.separable_conv2d(self.pconv6, None, 3, 1, 1,
                        normalizer_fn=self.normalizer, normalizer_params=self.bn_params, scope='Conv2d_{}_depthwise'.format(i))
        self.pconv71 = tc.layers.conv2d(self.dconv71, 512, 1, normalizer_fn=self.normalizer,
                        normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        # 7_2

        i += 1
        self.dconv72 = tc.layers.separable_conv2d(self.pconv71, None, 3, 1, 1,
                                                  normalizer_fn=self.normalizer, normalizer_params=self.bn_params,
                                                  scope='Conv2d_{}_depthwise'.format(i))
        self.pconv72 = tc.layers.conv2d(self.dconv72, 512, 1, normalizer_fn=self.normalizer,
                                        normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        # 7_3
        i += 1
        self.dconv73 = tc.layers.separable_conv2d(self.pconv72, None, 3, 1, 1,
                                                  normalizer_fn=self.normalizer, normalizer_params=self.bn_params,
                                                  scope='Conv2d_{}_depthwise'.format(i))
        self.pconv73 = tc.layers.conv2d(self.dconv73, 512, 1, normalizer_fn=self.normalizer,
                                        normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        # 7_4
        i += 1
        self.dconv74 = tc.layers.separable_conv2d(self.pconv73, None, 3, 1, 1,
                                                  normalizer_fn=self.normalizer, normalizer_params=self.bn_params,
                                                  scope='Conv2d_{}_depthwise'.format(i))
        self.pconv74 = tc.layers.conv2d(self.dconv74, 512, 1, normalizer_fn=self.normalizer,
                                        normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        # 7_5
        i += 1
        self.dconv75 = tc.layers.separable_conv2d(self.pconv74, None, 3, 1, 1,
                                                  normalizer_fn=self.normalizer, normalizer_params=self.bn_params,
                                                  scope='Conv2d_{}_depthwise'.format(i))
        self.pconv75 = tc.layers.conv2d(self.dconv75, 512, 1, normalizer_fn=self.normalizer,
                                        normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        # 8

        i += 1
        self.dconv8 = tc.layers.separable_conv2d(self.pconv75, None, 3, 1, 2,
                                normalizer_fn=self.normalizer, normalizer_params=self.bn_params,
                                                 scope='Conv2d_{}_depthwise'.format(i))
        self.pconv8 = tc.layers.conv2d(self.dconv8, 1024, 1, normalizer_fn=self.normalizer,
                                           normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        # 9

        i += 1
        self.dconv9 = tc.layers.separable_conv2d(self.pconv8, None, 3, 1, 1,
                                normalizer_fn=self.normalizer, normalizer_params=self.bn_params,
                                                 scope='Conv2d_{}_depthwise'.format(i))
        self.pconv9 = tc.layers.conv2d(self.dconv9, 1024, 1, normalizer_fn=self.normalizer,
                                           normalizer_params=self.bn_params, scope='Conv2d_{}_pointwise'.format(i))

        with tf.variable_scope('global_avg_pooling'):
            self.pool = tc.layers.avg_pool2d(self.pconv9, kernel_size=7, stride=1)
        with tf.variable_scope('Logits'):
            self.output = tc.layers.conv2d(self.pool, self.classnum, 1, activation_fn=None, scope='Conv2d_1c_1x1')
            shapes = self.output.get_shape().as_list()
            self.out = tf.reshape(self.output, [-1, shapes[1] * shapes[2] * shapes[3]])

        with tf.variable_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.out, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

"""
# Important for Batch Normalization
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate).minimize(self.loss)
"""

if __name__ == '__main__':
    "Test tranfer learing"
    model = MobileNetV1(True)
    img_dir = "a test picture path"
    raw_img  = cv2.imread(img_dir)
    #only update Logits layer
    exclude_vars = ['Logits']
    model_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    #print('the type of model_train_vars is {}'.format(type(model_train_vars)))
    reuse_vars = []
    train_vars = []
    for model_train_var in model_train_vars:
        excluede = False
        for exclude_var in exclude_vars:
            if exclude_var in model_train_var.name:
                excluede = True
                break

        if excluede:
            print('retrain tensor {}'.format(model_train_var.name))
            train_vars.append(model_train_var)
        else:
            print('reuse tensor {}'.format(model_train_var.name))
            reuse_vars.append(model_train_var)


    #print(model.output.get_shape())
    #print("The name of self.pool :",model.pool.op.name)
    #board_writer = tf.summary.FileWriter(logdir='./', graph=tf.get_default_graph())
    org_saver = tf.train.Saver(reuse_vars)
    new_saver = tf.train.Saver()
    fake_data = np.reshape(raw_img.astype(np.float32),(1,224,224,3))

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('The original initialized tensor of model.conv1 is:{}'.format(
            sess.run(model.conv1, feed_dict={model.input: fake_data})))

        check_point_file = 'mobilenet_v1_1.0_224_2017_06_14/mobilenet_v1_1.0_224.ckpt'
        tensor_name = 'MobilenetV1/Conv2d_0/BatchNorm/beta'
        org_saver.restore(sess, check_point_file)
        tensor_var = get_tensor.return_tensors_in_checkpoint_file(check_point_file, tensor_name=tensor_name,
                                             all_tensors=False, all_tensor_names=False)
        print('The tensor of {} in the original checkpoint is {}'.format(tensor_name, tensor_var))
        print('The new tensor of model.conv1 is:{}'.format(
            sess.run(model.conv1, feed_dict={model.input: fake_data})))

        new_saver.save(sess, './new_checkpoint/my_new_checkpoint.ckpt')
        tensor_var = get_tensor.return_tensors_in_checkpoint_file(check_point_file, tensor_name=tensor_name,
                                                                  all_tensors=False, all_tensor_names=False)
        print('The tensor of {} in the new checkpoint is {}'.format(tensor_name, tensor_var))

