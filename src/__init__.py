try:
    import tensorflow as tf
    TF_MODELS = True
except ModuleNotFoundError:
    import torch
    TF_MODELS = False