from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import sys
import numpy as np
import tensorflow as tf
import argparse
import os.path as osp
import cv2 as cv
from PIL import Image
from functools import partial
from IPython import embed

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


def cnn_model_fn(features, labels, mode, num_classes=20):
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])
    input_weights = tf.reshape(features["w"], [-1, 20])
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 64 * 64 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                            activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=num_classes)

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
    #onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.identity(tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits, weights = input_weights), name='loss')

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
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
    Argskvkcjkcgebvlkkkbfiekfjjcrurnn (str): Path to the VOC2007 directory.
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
    total_file = open("%s/ImageSets/Main/%s.txt" % (data_dir, split)).readlines()
    image_num = len(total_file)
    images = np.ndarray([image_num, 256, 256, 3], dtype = np.float32)
    labels = np.ndarray([image_num, 20], dtype = np.int32)
    weights = np.ndarray([image_num, 20], dtype = np.int32)
    for i in range(0, image_num):
        image_file = ("%s/JPEGImages/%s.jpg" % (data_dir, total_file[i].split()[0]))
        #print(image_file)
        image = cv.resize(cv.imread(image_file), (256, 256))
        images[i,:,:,:] = image
    for class_index in range(0,20):
        class_name = CLASS_NAMES[class_index]
        class_file = open("%s/ImageSets/Main/%s_%s.txt" % (data_dir, class_name, split)).readlines()
        for i in range(0, image_num):
            label = int(class_file[i].split()[1])
            labels[i, class_index] = 1 if label >=0 else 0
            weights[i, class_index] = 0 if label == 0 else 1
            #print("%d, %d" % (class_index, label))
    return images, labels ,weights       


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
    args = parse_args()
    BATCH_SIZE = 100
    NUM_ITERS = 10
    # Load training and eval data
    train_data, train_labels, train_weights = load_pascal(
        args.data_dir, split='trainval')
    eval_data, eval_labels, eval_weights = load_pascal(
        args.data_dir, split='test')

    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn,
                         num_classes=train_labels.shape[1]),
        model_dir="models/pascal_model_scratch")
    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data, "w": train_weights},
        y=train_labels,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True)
    sess = tf.Session()
    mAP = tf.Variable(initial_value=0, dtype=tf.float32)
    mydata = tf.summary.scalar("mAP", mAP)
    writer = tf.summary.FileWriter("models/mAP_scratch", sess.graph)
    init = tf.initialize_all_variables()
    sess.run(init)
    for i in range(0, 100):
        print(i)
        pascal_classifier.train(
            input_fn=train_input_fn,
            steps=NUM_ITERS,
            hooks=[logging_hook])
        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data, "w": eval_weights},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
        pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
        pred = np.stack([p['probabilities'] for p in pred])
        #rand_AP = compute_map(
        #    eval_labels, np.random.random(eval_labels.shape),
        #    eval_weights, average=None)
        #print('Random AP: {} mAP'.format(np.mean(rand_AP)))
        #gt_AP = compute_map(
        #    eval_labels, eval_labels, eval_weights, average=None)
        #print('GT AP: {} mAP'.format(np.mean(gt_AP)))
        AP = compute_map(eval_labels, pred, eval_weights, average=None)
        print('Obtained {} mAP'.format(np.mean(AP)))
        print('per class:')
        #for cid, cname in enumerate(CLASS_NAMES):
        #    print('{}: {}'.format(cname, _get_el(AP, cid)))
        update = tf.assign(mAP, np.mean(AP))
        #tf.summary.scalar("mAP", mAP)
        #merged = tf.summary.merge_all()
        #writer = tf.summary.FileWriter("models/pascal_model_scratch", sess.graph)
        sess.run(update)
        result = sess.run(mydata)
        writer.add_summary(result, i)

if __name__ == "__main__":
    main()
