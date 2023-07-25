try:
    import tensorflow as tf
    TF_MODELS = True
    IN_CHANNEL_DIM = -2
    OUT_CHANNEL_DIM = -1
except ModuleNotFoundError:
    import torch
    TF_MODELS = False
    IN_CHANNEL_DIM = 1
    OUT_CHANNEL_DIM = 0