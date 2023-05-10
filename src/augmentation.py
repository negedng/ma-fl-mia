import tensorflow as tf

def aug_crop(image, conf):
    p = conf["aug_crop"]
    paddings = tf.constant([[p, p], [p, p], [0,0]])
    orig_shape = image.shape
    image_p = tf.pad(image, paddings, "CONSTANT")
    image_c = tf.image.random_crop(image_p,orig_shape)
        
    return image
    
def aug_ds(image, label, conf):
    if conf["aug_crop"]>0:
        image = aug_crop(image, conf)
    return image, label
