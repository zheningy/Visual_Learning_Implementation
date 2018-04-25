from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import sys
import os
import numpy as np
import tensorflow as tf
import argparse
import os.path as osp
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt

from eval import compute_map
#import models

tf.logging.set_verbosity(tf.logging.INFO)

CLASS_NAMES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]

def name_in_checkpoint(v):
    return v.name.replace('kernel', 'weights').replace('bias', 'biases').replace('conv2d/', '').replace('dense/', '')[:-2]


def conv_layer(bottom, num_output, k_size, padding='same'):
    next_layer = tf.layers.conv2d(
        inputs=bottom,
        filters=num_output,
        kernel_size=[k_size,k_size],
        padding=padding,
        activation=tf.nn.relu,
        bias_initializer=tf.zeros_initializer())
    return next_layer


def cnn_model_fn(features, labels, mode, num_classes=20):
    # Write this function
    """Model function for CNN."""
    # Input Layer

    input_layer = tf.reshape(features["x"], [-1, 224, 224, 3])
    input_weights = tf.reshape(features["w"], [-1, 20])
    with tf.variable_scope('vgg_16'):
        # Convolutional Layer #1_1, 1_2
        with tf.variable_scope('conv1'):
            with tf.variable_scope('conv1_1'):
                conv1_1 = conv_layer(input_layer, 64, 3)
            with tf.variable_scope('conv1_2'):
                conv1_2 = conv_layer(conv1_1, 64, 3)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2_1, 2_2
        with tf.variable_scope('conv2'):
            with tf.variable_scope('conv2_1'):
                conv2_1 = conv_layer(pool1, 128, 3)
            with tf.variable_scope('conv2_2'):
                conv2_2 = conv_layer(conv2_1, 128, 3)

        # Pooling Layer #2
        pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=2)

        # Convolutional Layer #3_1, 3_2, 3_3
        with tf.variable_scope('conv3'):
            with tf.variable_scope('conv3_1'):
                conv3_1 = conv_layer(pool2, 256, 3)
            with tf.variable_scope('conv3_2'):
                conv3_2 = conv_layer(conv3_1, 256, 3)
            with tf.variable_scope('conv3_3'):
                conv3_3 = conv_layer(conv3_2, 256, 3)

        # Pooling Layer #3
        pool3 = tf.layers.max_pooling2d(inputs=conv3_3, pool_size=[2, 2], strides=2)

        # Convolutional Layer #4_1, 4_2, 4_3
        with tf.variable_scope('conv4'):
            with tf.variable_scope('conv4_1'):
                conv4_1 = conv_layer(pool3, 512, 3)
            with tf.variable_scope('conv4_2'):
                conv4_2 = conv_layer(conv4_1, 512, 3)
            with tf.variable_scope('conv4_3'):
                conv4_3 = conv_layer(conv4_2, 512, 3)

        # Pooling Layer #4
        pool4 = tf.layers.max_pooling2d(inputs=conv4_3, pool_size=[2, 2], strides=2)

        # Convolutional Layer #5_1, 5_2, 5_3
        with tf.variable_scope('conv5'):
            with tf.variable_scope('conv5_1'):
                conv5_1 = conv_layer(pool4, 512, 3)
            with tf.variable_scope('conv5_2'):
                conv5_2 = conv_layer(conv5_1, 512, 3)
            with tf.variable_scope('conv5_3'):
                conv5_3 = conv_layer(conv5_2, 512, 3)

        # Pooling Layer #5
        pool5 = tf.layers.max_pooling2d(inputs=conv5_3, pool_size=[2, 2], strides=2)

        pool5_flat = tf.reshape(pool5, [-1, 7 * 7 * 512])
        # Dense layer #1 and dropout
        with tf.variable_scope('fc6'):
            fc6 = tf.layers.dense(inputs=pool5_flat, units=4096,
                                    activation=tf.nn.relu,
                                    bias_initializer=tf.zeros_initializer())
        dropout1 = tf.layers.dropout(
            inputs=fc6, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

        # Dense layer #2 and dropout
        with tf.variable_scope('fc7'):
            fc7 = tf.layers.dense(inputs=dropout1, units=4096,
                                    activation=tf.nn.relu,
                                    bias_initializer=tf.zeros_initializer())
        dropout2 = tf.layers.dropout(
            inputs=fc7, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

        # Dense layer #2 and dropout
        with tf.variable_scope('fc8'):
            fc8 = tf.layers.dense(inputs=dropout2, units=1000,
                                    activation=tf.nn.relu,
                                    bias_initializer=tf.zeros_initializer(),
                                    kernel_initializer=tf.truncated_normal_initializer(0, 0.01))
        dropout3 = tf.layers.dropout(
            inputs=fc8, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout3, units=20,
                                bias_initializer=tf.zeros_initializer(),
                                kernel_initializer=tf.truncated_normal_initializer(0, 0.01))

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.to_int32(logits>0.5),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.sigmoid(logits, name="sigmoid_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.identity(tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits, weights = input_weights), name='loss')

    starter_learning_rate = 0.0001
    global_step = tf.train.get_global_step()
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000, 0.5, staircase=True)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        vars_to_restore = []
        for var in tf.trainable_variables():
            if ('conv' in var.name or 'fc6' in var.name or 'fc7' in var.name) and 'Momentum' not in var.name:
                vars_to_restore.append(var)
                # print(var.name)
                print(name_in_checkpoint(var), var.shape)
        vars_to_restore = {name_in_checkpoint(v): v for v in vars_to_restore}
        loader = tf.train.Saver(vars_to_restore, reshape=True)

        def init_fn(scaffold, session):
            loader.restore(session, 'vgg_16.ckpt')

        scaffold = tf.train.Scaffold(init_fn=init_fn)

        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op, scaffold=scaffold)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def load_pascal(data_dir, split='train'):
    """
    Function to read images from PASCAL data folder.
    Args:
        data_dir (str): Path to the VOC2007 directory.
        split (str): train/val/trainval split to use.
    Returns:
        images (np.ndarray): Return a np.float32 array of
            shape (N, H, W, 3), where H, W are 224px each,
            and each image is in RGB format.
        labels (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are active in that image.
    """
    # Wrote this function
    img_dir = data_dir + "/JPEGImages"
    label_dir = data_dir + "/ImageSets/Main"

    total_file = open(label_dir + '/' + split + '.txt').readlines()

    image_num = len(total_file)
    # construct images
    images = np.zeros((image_num, 224, 224, 3), dtype = np.float32)
    for i in range(image_num):
        file_name = total_file[i].split()[0]
        img = Image.open(img_dir + '/' + file_name + '.jpg')
        img = img.resize((224,224))
        images[i] = np.asarray(img, dtype = np.float32)

    # construct label and weights
    num_class = len(CLASS_NAMES)
    labels = np.zeros((image_num, num_class), dtype = np.int32)
    weights = np.ones((image_num, num_class), dtype = np.int32)

    for i in range(num_class):
        obj_dir = label_dir + '/' + CLASS_NAMES[i] + '_' + split + '.txt'
        f = open(obj_dir, 'r')
        index = 0
        for line in f.readlines():
            mark = line.split()[1].strip()
            if (mark == "1"):
                labels[index][i] = 1
            elif (mark == "0"):
                weights[index][i] = 0
            index += 1

    return images, labels, weights

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a classifier in tensorflow!')
    parser.add_argument(
        'data_dir', type=str, default='data/VOC2007',
        help='Path to PASCAL data storage')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def _get_el(arr, i):
    try:
        return arr[i]
    except IndexError:
        return arr


def main():
    BATCH_SIZE = 10
    NUM_ITERS = 1000
    args = parse_args()
    # Load training and eval data
    train_data, train_labels, train_weights = load_pascal(
        args.data_dir, split='trainval')
    eval_data, eval_labels, eval_weights = load_pascal(
        args.data_dir, split='test')

    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn,
                         num_classes=train_labels.shape[1]),
        model_dir="models/pascal_model_VGGFine")
    tensors_to_log = {"loss": "loss"}


    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data, "w": train_weights},
        y=train_labels,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True)


    loss_list = []
    acc_list = []
    record_steps = 100

    # for i in range(record_steps):
    #     pascal_classifier.train(
    #         input_fn=train_input_fn,
    #         steps=NUM_ITERS/record_steps,
    #         hooks=[logging_hook])
    #     # Evaluate the model and print results
    #     eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #         x={"x": eval_data, "w": eval_weights},
    #         y=eval_labels,
    #         batch_size=1,
    #         num_epochs=1,
    #         shuffle=False)
    #     pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
    #     pred = np.stack([p['probabilities'] for p in pred])
    #     # rand_AP = compute_map(
    #     #     eval_labels, np.random.random(eval_labels.shape),
    #     #     eval_weights, average=None)
    #     # print('Random AP: {} mAP'.format(np.mean(rand_AP)))
    #     # gt_AP = compute_map(
    #     #     eval_labels, eval_labels, eval_weights, average=None)
    #     # print('GT AP: {} mAP'.format(np.mean(gt_AP)))
    #     AP = compute_map(eval_labels, pred, eval_weights, average=None)
    #     print('Obtained {} mAP'.format(np.mean(AP)))
    #     # print('per class:')
    #     # for cid, cname in enumerate(CLASS_NAMES):
    #     #     print('{}: {}'.format(cname, _get_el(AP, cid)))
    #
    #     acc_list.append(np.mean(AP))
    #
    # mAPfile = open('mAP_Q4.txt', 'w')
    # for item in acc_list:
    #   mAPfile.write("%s\n" % item)


    pascal_classifier.train(
        input_fn=train_input_fn,
        steps=NUM_ITERS,
        hooks=[logging_hook])
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data, "w": eval_weights},
        y=eval_labels,
        batch_size=1,
        num_epochs=1,
        shuffle=False)
    pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
    pred = np.stack([p['probabilities'] for p in pred])
    # rand_AP = compute_map(
    #     eval_labels, np.random.random(eval_labels.shape),
    #     eval_weights, average=None)
    # print('Random AP: {} mAP'.format(np.mean(rand_AP)))
    # gt_AP = compute_map(
    #     eval_labels, eval_labels, eval_weights, average=None)
    # print('GT AP: {} mAP'.format(np.mean(gt_AP)))
    AP = compute_map(eval_labels, pred, eval_weights, average=None)
    print('Obtained {} mAP'.format(np.mean(AP)))
    # print('per class:')
    # for cid, cname in enumerate(CLASS_NAMES):
    #     print('{}: {}'.format(cname, _get_el(AP, cid)))


if __name__ == "__main__":
    main()
