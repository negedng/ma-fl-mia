import tensorflow as tf
import numpy as np

from src.models.tensorflow_models.layer_utils import get_scaler, get_norm

kaiming_normal = tf.keras.initializers.VarianceScaling(
    scale=2.0, mode="fan_out", distribution="untruncated_normal"
)


def basic_block(
    x,
    planes,
    stride=1,
    name=None,
    use_scaler=False,
    scaler_rate=1.0,
    keep_scaling=False,
    norm_mode="bn",
    static_bn=False,
):
    downsample_out = x

    out = x
    out = get_scaler(use_scaler, scaler_rate, keep_scaling, name=f"{name}.scaler1")(out)
    out = get_norm(norm_mode, static_bn, name=f"{name}.bn1")(out)
    out = tf.keras.layers.ReLU(name=f"{name}.relu1")(out)

    downsample = None
    inplanes = x.shape[3]
    if stride != 1 or inplanes != planes:
        downsample = [
            tf.keras.layers.Conv2D(
                filters=planes,
                kernel_size=1,
                strides=stride,
                use_bias=False,
                kernel_initializer=kaiming_normal,
                name=f"{name}.0.downsample.0",
            ),
        ]
        downsample_out = out
        for layer in downsample:
            downsample_out = layer(downsample_out)

        

    out = tf.keras.layers.Conv2D(
        filters=planes,
        kernel_size=3,
        strides=stride,
        use_bias=False,
        kernel_initializer=kaiming_normal,
        name=f"{name}.conv1",
        padding="same",
    )(out)


    out = get_scaler(use_scaler, scaler_rate, keep_scaling, name=f"{name}.scaler2")(out)
    out = get_norm("bn", False, name=f"{name}.bn2")(out)
    out = tf.keras.layers.ReLU(name=f"{name}.relu2")(out)
    out = tf.keras.layers.Conv2D(
        filters=planes,
        kernel_size=3,
        strides=1,
        use_bias=False,
        kernel_initializer=kaiming_normal,
        name=f"{name}.conv2",
        padding="same",
    )(out)

    out = tf.keras.layers.Add(name=f"{name}.add")([downsample_out, out])
    
    return out


def make_layer(
    x,
    planes,
    blocks,
    stride=1,
    name=None,
    use_scaler=False,
    scaler_rate=1.0,
    keep_scaling=False,
    norm_mode="bn",
    static_bn=False,
):

    x = basic_block(
        x,
        planes,
        stride,
        name=f"{name}.0",
        use_scaler=use_scaler,
        scaler_rate=scaler_rate,
        keep_scaling=keep_scaling,
        norm_mode=norm_mode,
        static_bn=static_bn,
    )
    for i in range(1, blocks):
        x = basic_block(
            x,
            planes,
            name=f"{name}.{i}",
            use_scaler=use_scaler,
            scaler_rate=scaler_rate,
            keep_scaling=keep_scaling,
            norm_mode=norm_mode,
            static_bn=static_bn,
        )

    return x


def resnet(
    blocks_per_layer,
    unit_size=64,
    num_classes=10,
    input_shape=(32, 32, 3),
    use_scaler=True,
    model_rate=1.0,
    keep_scaling=False,
    norm_mode="bn",
    static_bn=False,
):
    #!TODO: there is a problem with scaling
    scaler_rate = model_rate

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    # small images have different ResNet: https://github.com/raphaelreme/torch-resnet/blob/main/torch_resnet/resnet.py
    if max(input_shape) < 100:
        x = tf.keras.layers.Conv2D(
            filters=unit_size,
            kernel_size=3,
            strides=1,
            use_bias=False,
            kernel_initializer=kaiming_normal,
            padding="same",
            name="conv1",
        )(x)
    else:
        x = tf.keras.layers.Conv2D(
            filters=unit_size,
            kernel_size=7,
            strides=2,
            use_bias=False,
            kernel_initializer=kaiming_normal,
            padding="same",
            name="conv1",
        )(x)

        x = tf.keras.layers.MaxPool2D(
            pool_size=3, strides=2, padding="same", name="maxpool"
        )(x)

    x = make_layer(x, unit_size, blocks_per_layer[0], name="layer1")
    x = make_layer(x, unit_size * 2, blocks_per_layer[1], stride=2, name="layer2")
    x = make_layer(x, unit_size * 4, blocks_per_layer[2], stride=2, name="layer3")
    x = make_layer(x, unit_size * 8, blocks_per_layer[3], stride=2, name="layer4")

    x = get_scaler(use_scaler, scaler_rate, keep_scaling, name=f"scaler1")(x)
    x = get_norm(norm_mode, static_bn, name=f"bn1")(x)
    x = tf.keras.layers.ReLU(name="relu1")(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="avgpool")(x)
    initializer = tf.keras.initializers.RandomUniform(
        -1.0 / np.sqrt(512), 1.0 / np.sqrt(512)
    )
    x = tf.keras.layers.Dense(
        units=num_classes,
        kernel_initializer=initializer,
        bias_initializer=initializer,
        name="fc",
    )(x)
    outputs = x

    model = tf.keras.Model(inputs, outputs)
    return model


def resnet18(**kwargs):
    return resnet([2, 2, 2, 2], **kwargs)
