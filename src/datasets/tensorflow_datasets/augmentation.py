# Default parameters to match: https://github.com/diaoenmao/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients/blob/master/src/data.py
# and https://github.com/bearpaw/pytorch-classification/blob/master/cifar.py

import tensorflow as tf


def aug_crop(image, conf):
    p = conf["aug_crop"]
    paddings = tf.constant([[p, p], [p, p], [0, 0]])
    orig_shape = image.shape
    image_p = tf.pad(image, paddings, "CONSTANT")
    image_c = tf.image.random_crop(image_p, orig_shape)

    return image_c


def aug_horizontal_flip(image):
    return tf.image.random_flip_left_right(image)


def aug_ds(image, label, conf):
    if conf["aug_crop"] > 0:
        image = aug_crop(image, conf)
    if conf["aug_horizontal_flip"]:
        image = aug_horizontal_flip(image)
    return image, label
