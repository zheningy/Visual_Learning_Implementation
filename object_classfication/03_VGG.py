from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import sys
import os
from sets import Set
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

def conv_layer(bottom, num_output, k_size, padding='same'):
    next_layer = tf.layers.conv2d(
        inputs=bottom,
        filters=num_output,
        kernel_size=[k_size,k_size],
        padding=padding,
        activation=tf.nn.relu,
        bias_initializer=tf.zeros_initializer())
    return next_layer

# # Thanks Tao Chen for providing this function
# from tensorflow.core.framework import summary_pb2
# def summary_var(log_dir, name, val, step):
#     writer = tf.summary.FileWriterCache.get(log_dir)
#     summary_proto = summary_pb2.Summary()
#     value = summary_proto.value.add()
#     value.tag = name
#     value.simple_value = float(val)
#     writer.add_summary(summary_proto, step)
#     writer.flush()

def cnn_model_fn(features, labels, mode, num_classes=20):
    # Write this function
    """Model function for CNN."""
    # Input Layer
    #flip_imgs = tf.map_fn(lambda img: tf.image.random_flip_left_right(img, 0.5), features["x"])
    input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])
    input_weights = tf.reshape(features["w"], [-1, 20])
    if mode == tf.estimator.ModeKeys.TRAIN:
        input_layer = tf.random_crop(input_layer,[input_layer.shape[0],227,227,3])
        input_layer = tf.stack([tf.image.random_flip_left_right(input_layer[i])for i in range(input_layer.shape[0])])
    if mode == tf.estimator.ModeKeys.PREDICT:
        input_layer = tf.image.resize_image_with_crop_or_pad(input_layer, 227, 227)

    

    # Convolutional Layer #1_1, 1_2
    conv1_1 = conv_layer(input_layer, 64, 3)
    conv1_2 = conv_layer(conv1_1, 64, 3)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2_1, 2_2
    conv2_1 = conv_layer(pool1, 128, 3)
    conv2_2 = conv_layer(conv2_1, 128, 3)

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3_1, 3_2, 3_3
    conv3_1 = conv_layer(pool2, 256, 3)
    conv3_2 = conv_layer(conv3_1, 256, 3)  
    conv3_3 = conv_layer(conv3_2, 256, 3)

    # Pooling Layer #3
    pool3 = tf.layers.max_pooling2d(inputs=conv3_3, pool_size=[2, 2], strides=2)

    # Convolutional Layer #4_1, 4_2, 4_3
    conv4_1 = conv_layer(pool3, 512, 3)
    conv4_2 = conv_layer(conv4_1, 512, 3)  
    conv4_3 = conv_layer(conv4_2, 512, 3)

    # Pooling Layer #4
    pool4 = tf.layers.max_pooling2d(inputs=conv4_3, pool_size=[2, 2], strides=2)

    # Convolutional Layer #5_1, 5_2, 5_3
    conv5_1 = conv_layer(pool4, 512, 3)
    conv5_2 = conv_layer(conv5_1, 512, 3)  
    conv5_3 = conv_layer(conv5_2, 512, 3)

    # Pooling Layer #5
    pool5 = tf.layers.max_pooling2d(inputs=conv5_3, pool_size=[2, 2], strides=2)

    pool5_flat = tf.reshape(pool5, [-1, 7 * 7 * 512])
    # Dense layer #1 and dropout
    fc6 = tf.layers.dense(inputs=pool5_flat, units=4096,
                            activation=tf.nn.relu,
                            bias_initializer=tf.zeros_initializer())
    dropout1 = tf.layers.dropout(
        inputs=fc6, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Dense layer #2 and dropout
    fc7 = tf.layers.dense(inputs=dropout1, units=4096,
                            activation=tf.nn.relu,
                            bias_initializer=tf.zeros_initializer())
    dropout2 = tf.layers.dropout(
        inputs=fc7, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Dense layer #2 and dropout
    fc8 = tf.layers.dense(inputs=dropout2, units=1000,
                            activation=tf.nn.relu,
                            bias_initializer=tf.zeros_initializer())
    dropout3 = tf.layers.dropout(
        inputs=fc8, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout3, units=num_classes)




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


    starter_learning_rate = 0.001
    global_step = tf.train.get_global_step()
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           10000, 0.5, staircase=True, name='learning_rate')

    tf.summary.scalar("learning_rate", learning_rate)
    tf.summary.image("input_layer",input_layer)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        train_op = tf.contrib.layers.optimize_loss(
              loss, global_step, learning_rate=learning_rate, optimizer=optimizer,
              summaries=["gradients"])
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

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
    images = np.zeros((image_num, 256, 256, 3), dtype = np.float32)
    for i in range(image_num):
        file_name = total_file[i].split()[0]
        img = Image.open(img_dir + '/' + file_name + '.jpg')
        img = img.resize((256, 256))
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
    NUM_ITERS = 40000
    args = parse_args()
    # Load training and eval data
    train_data, train_labels, train_weights = load_pascal(
        args.data_dir, split='trainval')
    eval_data, eval_labels, eval_weights = load_pascal(
        args.data_dir, split='test')

    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn,
                         num_classes=train_labels.shape[1]),
        model_dir="models/pascal_model_VGGNet")
    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=500)
    # summary_hook = tf.train.SummarySaverHook(save_steps=200, 
    #                 output_dir="models/VGGNet_image_lr", 
    #                 summary_op=tf.summary.merge_all())
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

    for i in xrange(record_steps):
        pascal_classifier.train(
            input_fn=train_input_fn,
            steps=NUM_ITERS/record_steps,
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

        acc_list.append(np.mean(AP))

    mAPfile = open('mAP_Q3.txt', 'w')
    for item in acc_list:
      mAPfile.write("%s\n" % item)


    # pascal_classifier.train(
    #     input_fn=train_input_fn,
    #     steps=NUM_ITERS,
    #     hooks=[logging_hook])
    # # Evaluate the model and print results
    # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": eval_data, "w": eval_weights},
    #     y=eval_labels,
    #     batch_size=1,
    #     num_epochs=1,
    #     shuffle=False)
    # pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
    # pred = np.stack([p['probabilities'] for p in pred])
    # # rand_AP = compute_map(
    # #     eval_labels, np.random.random(eval_labels.shape),
    # #     eval_weights, average=None)
    # # print('Random AP: {} mAP'.format(np.mean(rand_AP)))
    # # gt_AP = compute_map(
    # #     eval_labels, eval_labels, eval_weights, average=None)
    # # print('GT AP: {} mAP'.format(np.mean(gt_AP)))
    # AP = compute_map(eval_labels, pred, eval_weights, average=None)
    # print('Obtained {} mAP'.format(np.mean(AP)))
    # # print('per class:')
    # # for cid, cname in enumerate(CLASS_NAMES):
    # #     print('{}: {}'.format(cname, _get_el(AP, cid)))


if __name__ == "__main__":
    main()