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
import scipy.misc

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


# def load_target_images(dir):
#     images_name = ('000031.jpg', '000037.jpg', '000119.jpg', '000152.jpg', '000176.jpg',
#                     '000178.jpg', '000201.jpg', '000202.jpg', '000383.jpg', '000521.jpg')



def cnn_model_fn(features, labels, mode, num_classes=20):
    # Write this function
    """Model function for CNN."""
    # Input Layer
    # input_layer = None
    # input_weights = tf.reshape(features["w"], [-1, 20])

    # batch_size = features["x"].shape[0]
    # if mode == tf.estimator.ModeKeys.PREDICT:
    #     central_crops = tf.map_fn(lambda img: tf.image.central_crop(img, 0.8), features["x"])
    #     input_layer = tf.image.resize_images(central_crops,[256, 256])
    # else:
    #     flip_imgs = tf.map_fn(lambda img: tf.image.random_flip_left_right(img, 0.5), features["x"])
    #     rand_crops = tf.map_fn(lambda img: tf.random_crop(img, [200, 200, 3]), flip_imgs)
    #     input_layer = tf.image.resize_images(rand_crops,[256, 256])

    input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])
    input_weights = tf.reshape(features["w"], [-1, 20])

    if mode == tf.estimator.ModeKeys.TRAIN:
        input_layer = tf.random_crop(input_layer,[input_layer.shape[0],227,227,3])
        input_layer = tf.stack([tf.image.random_flip_left_right(input_layer[i])for i in range(input_layer.shape[0])])
    if mode == tf.estimator.ModeKeys.PREDICT:
        input_layer = tf.image.resize_image_with_crop_or_pad(input_layer, 227, 227)


    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        kernel_size=[11,11],
        strides=(4, 4),
        padding="valid",
        activation=tf.nn.relu,
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.truncated_normal_initializer(0, 0.01),
        name='conv1')

    conv1_kernel = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1/kernel')[0]

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=256,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.truncated_normal_initializer(0, 0.01))
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=384,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.truncated_normal_initializer(0, 0.01))

    # Convolutional Layer #4
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=384,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.truncated_normal_initializer(0, 0.01))

    # Convolutional Layer #5 and Pooling Layer #3
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.truncated_normal_initializer(0, 0.01))
    pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)

    # Dense Layer #1 and dropout
    pool3_flat = tf.reshape(pool3, [-1, 6 * 6 * 256])
    dense1 = tf.layers.dense(inputs=pool3_flat, units=4096,
                            activation=tf.nn.relu,
                            bias_initializer=tf.zeros_initializer(),
                            kernel_initializer=tf.truncated_normal_initializer(0, 0.01))
    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Dense Layer #2 and dropout
    dense2 = tf.layers.dense(inputs=dropout1, units=4096,
                            activation=tf.nn.relu,
                            bias_initializer=tf.zeros_initializer(),
                            kernel_initializer=tf.truncated_normal_initializer(0, 0.01))
    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout2, units=20,
                            bias_initializer=tf.zeros_initializer(),
                            kernel_initializer=tf.truncated_normal_initializer(0, 0.01))

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.to_int32(logits>0.5),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.sigmoid(logits, name="sigmoid_tensor"),
        "pool5_features": pool3_flat,
        "fc7_features": dense2,
        "input_image":input_layer
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.identity(tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits, weights = input_weights), name='loss')

    starter_learning_rate = 0.001
    global_step = tf.train.get_global_step()
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           10000, 0.5, staircase=True)


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
        img = img.resize((256,256))
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
    NUM_ITERS = 30000
    args = parse_args()
    # Load training and eval data
    train_data, train_labels, train_weights = load_pascal(
        args.data_dir, split='trainval')
    eval_data, eval_labels, eval_weights = load_pascal(
        args.data_dir, split='test')

    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn,
                         num_classes=20),
        model_dir="models/pascal_model_alexNet_neibour")
    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=500)
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data, "w": train_weights},
        y=train_labels,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True)


    loss_list = []
    acc_list = []
    record_steps = 1


    # pascal_classifier.train(
    #     input_fn=train_input_fn,
    #     steps=NUM_ITERS,
    #     hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data, "w": eval_weights},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    pred = list(pascal_classifier.predict(input_fn=eval_input_fn))

    target_img_index = (263, 191, 87, 77, 63, 18, 100, 101, 86, 17)

    input_images = np.stack([p['input_image'] for p in pred])

    for i in range(len(target_img_index)):
        curt_index = target_img_index[i]
        img = input_images[curt_index]
        scipy.misc.imsave("analysis/alex_img/%d_target_%d.png"%(i,curt_index), img)



    pool5_feature = np.stack([p['pool5_features'] for p in pred])
    
    
    nearest_img_index = [-1] * 10
    dist_img = [100000000] * 10

    for index in range(len(pred)):
        for j in range(10):
            target_index = target_img_index[j]
            if index == target_img_index[j]:
                continue
            else:
                curt_dist = np.linalg.norm(pool5_feature[index] - pool5_feature[target_index])
                if curt_dist < dist_img[j]:
                    nearest_img_index[j] = index
                    dist_img[j] = curt_dist
    print("pool5 match: ")
    print(nearest_img_index)

    for i in range(len(nearest_img_index)):
        curt_index = nearest_img_index[i]
        img = input_images[curt_index]
        scipy.misc.imsave("analysis/alex_img/%d_pool5_%d.png"%(i,curt_index), img)

    fc7_feature = np.stack([p['fc7_features'] for p in pred])
    nearest_img_index = [-1] * 10
    dist_img = [100000000] * 10
    print(fc7_feature[0].shape)
    for index in range(len(pred)):
        for j in range(10):
            target_index = target_img_index[j]
            if index == target_img_index[j]:
                continue
            else:
                curt_dist = np.linalg.norm(fc7_feature[index] - fc7_feature[target_index])
                if curt_dist < dist_img[j]:
                    nearest_img_index[j] = index
                    dist_img[j] = curt_dist
    print("fc7 match: ")
    print(nearest_img_index)

    for i in range(len(nearest_img_index)):
        curt_index = nearest_img_index[i]
        img = input_images[curt_index]
        scipy.misc.imsave("analysis/alex_img/%d_fc7_%d.png"%(i,curt_index), img)


    # fp = open('alex_pool5_feat.txt', 'w')
    # for i in range(len(pred)):
    #     fp.write("%s\n" % pred[i]['pool5_features'])




if __name__ == "__main__":
    main()