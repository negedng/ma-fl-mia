import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import os


def get_ds_from_np(data):
    return tf.data.Dataset.from_tensor_slices(data)


def get_np_from_ds(ds):
    try:
        ds = ds.unbatch()
    except ValueError:
        # already unbatched
        pass
    X = []
    Y = []
    for x, y in ds.as_numpy_iterator():
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)


def preprocess(image, conf):
    input_shape = (32, 32, 3)
    if image.shape != input_shape:
        image = tf.image.resize(image, input_shape[:2])
    if conf["data_normalize"]:
        # Convert to float32 and scale pixel values to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        # Subtract mean RGB values
    if conf["data_centralize"]:
        mean_rgb = tf.constant([0.491, 0.482, 0.447])
        std_rgb = tf.constant([0.247, 0.243, 0.261])
        image = (image - mean_rgb) / std_rgb
    return image


def preprocess_ds(image, label, conf):
    image = preprocess(image, conf)
    return image, label


def preprocess_data(data, conf, shuffle=False, cache=False):
    data = data.map(lambda x, y: preprocess_ds(x, y, conf))
    if shuffle:
        if cache:
            data = data.cache()
        data = data.shuffle(5000)
    data = data.batch(conf["batch_size"])
    if not shuffle and cache:
        data = data.cache()
    data = data.prefetch(tf.data.AUTOTUNE)
    return data


def load_data(
    dataset_mode="cifar10", val_split=True, conf={}
):
    if "val_split" in conf.keys():
        val_split = conf["val_split"]

    if dataset_mode == "cifar10":
        if val_split:
            train_ds, val_ds, test_ds = tfds.load(
                "cifar10",
                split=["train[5%:]", "train[:5%]", "test"],
                as_supervised=True,
            )
        else:
            train_ds, val_ds, test_ds = tfds.load("cifar10", split=["train", "test", "test"], as_supervised=True)

    return train_ds, val_ds, test_ds
