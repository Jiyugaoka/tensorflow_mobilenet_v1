#!/usr/bin/env python
# -*- coding:utf-8 -*-
#auther:jf183
#datetime:2018/8/31 9:31

# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import time
import datetime
from loaddata import Dataset
from Model import MobileNetV1

# Parameters
# ==================================================
tf.flags.DEFINE_string('train_anno_dir', '/root/DL/carDetection12_29/BIT_dataset/train_anno/', 'the directory of the train_anno')
tf.flags.DEFINE_string('test_anno_dir', '/root/DL/carDetection12_29/BIT_dataset/val_anno/', 'the directory of the val_anno')
tf.flags.DEFINE_string('output_dir', '/root/DL/carDetection12_29/', 'the directory of the dataset')
tf.flags.DEFINE_string('train_dataset_mean', '107.94, 115.34, 118.34', 'the piexl mean of your train dataset')
tf.flags.DEFINE_string('check_point_file', '/root/DL/carDetection12_29/mobilenet_v1_1.0_224_2017_06_14/mobilenet_v1_1.0_224.ckpt', 'the directory of the check_point')
tf.flags.DEFINE_string('exclude_vars', 'Logits, Conv2d_13, Conv2d_12', 'the vars to retrain')

# Training parameters
tf.flags.DEFINE_integer("classes", 6, "the number of classes")
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 30)")
tf.flags.DEFINE_integer("num_epochs", 80, "Number of training epochs (default: 20)")
tf.flags.DEFINE_integer("evaluate_every", 30, "Evaluate model on dev set after this many steps (default: 20)")
tf.flags.DEFINE_integer("checkpoint_every", 30, "Save model after this many steps (default: 20)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]

def main(_):

    # Load data
    train_dataset_mean = parse_comma_list(FLAGS.train_dataset_mean)
    dataseter = Dataset(FLAGS.classes, train_dataset_mean=train_dataset_mean)
    train_image_dir_list, train_label_list = dataseter.label_shuffle(FLAGS.train_anno_dir)
    #train_x, train_y = loaddata.read_data_label(train_image_dir_list, train_label_list)
    #train_y = [item[0] for item in train_y]

    test_image_dir_list, test_label_list = dataseter.get_input(FLAGS.test_anno_dir)
    test_x, test_y = dataseter.read_data_label(test_image_dir_list, test_label_list)
    test_y_label_num = [item[0] for item in test_y]
    

    print("Train/Test: {:d}/{:d}".format(len(train_image_dir_list), len(test_x)))

    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = MobileNetV1()
            exclude_vars = parse_comma_list(FLAGS.exclude_vars)
            model_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
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

            # Define Training procedure
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)
                grads_and_vars = optimizer.compute_gradients(model.loss, var_list=train_vars)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.join(FLAGS.output_dir, "runs_181229", timestamp)
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Test summaries
            test_summary_op = tf.summary.merge([loss_summary, acc_summary])
            test_summary_dir = os.path.join(out_dir, "summaries", "test")
            test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                
            org_saver = tf.train.Saver(reuse_vars)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            org_saver.restore(sess, FLAGS.check_point_file)

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_y: y_batch,
                    model.is_training: True
                    #model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def test_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_y: y_batch,
                    model.is_training: False
                    #model.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, test_summary_op, model.loss, model.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = dataseter.batch_iter(train_image_dir_list, train_label_list, FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
            #
                x_batch, y_batch = batch
                y_batch_label_num = [items[0] for items in y_batch]
                train_step(x_batch, y_batch_label_num)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    test_step(test_x, test_y_label_num, writer=test_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

if __name__ == '__main__':
    tf.app.run()


