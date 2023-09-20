from src import TF_MODELS

if TF_MODELS:
    from .tensorflow_models.model_utils import get_optimizer, get_loss
    from .tensorflow_models.model_utils import predict
    from .tensorflow_models.model_utils import predict_losses
    from .tensorflow_models.model_utils import evaluate
    from .tensorflow_models.model_utils import fit
    from .tensorflow_models.model_utils import set_weights, load_model_weights
    from .tensorflow_models.model_utils import get_weights
    from .tensorflow_models.model_utils import save_model
    from .tensorflow_models.model_utils import print_summary
    from .tensorflow_models.model_utils import calculate_loss
    from .tensorflow_models.model_utils import count_params
    from .tensorflow_models.model_utils import prepare_model
    
    from .tensorflow_models.diao_cnn import get_diao_CNN
    from .tensorflow_models.alexnet import get_alexnet
    from .tensorflow_models.resnet import get_resnet18
    from .tensorflow_models.simple_cnn import get_simple_CNN
else:
    from .pytorch_models.model_utils import get_optimizer, get_loss
    from .pytorch_models.model_utils import get_weights, set_weights, load_model_weights
    from .pytorch_models.model_utils import prepare_model
    from .pytorch_models.model_utils import evaluate
    from .pytorch_models.model_utils import save_model
    from .pytorch_models.model_utils import print_summary
    from .pytorch_models.model_utils import fit, predict, predict_losses
    from .pytorch_models.model_utils import calculate_loss
    from .pytorch_models.model_utils import count_params
    from .pytorch_models.model_utils import get_gradients

    from .pytorch_models.diao_cnn import get_diao_CNN
    from .pytorch_models.resnet import get_resnet18
    def get_alexnet(*args, **kwargs):
        raise NotImplementedError("alexnet not implemented using PyTorch")
    def get_simple_CNN(*args, **kwargs):
        raise NotImplementedError("simple_CNN not implemented using PyTorch")