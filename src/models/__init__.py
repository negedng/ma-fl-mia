from src import TF_MODELS

if TF_MODELS:
    from .tensorflow_models.model_utils import get_model_architecture
    from .tensorflow_models.model_utils import get_optimizer, get_loss
    from .tensorflow_models.model_utils import init_model
    from .tensorflow_models.model_utils import predict
    from .tensorflow_models.model_utils import predict_losses
    from .tensorflow_models.model_utils import evaluate
    from .tensorflow_models.model_utils import fit
    from .tensorflow_models.model_utils import set_weights
    from .tensorflow_models.model_utils import get_weights
    from .tensorflow_models.model_utils import save_model
    from .tensorflow_models.model_utils import print_summary
    from .tensorflow_models.model_utils import calculate_loss
    from .tensorflow_models.model_utils import count_params
else:
    from .pytorch_models.model_utils import get_optimizer, get_loss
    from .pytorch_models.model_utils import get_weights, set_weights
    from .pytorch_models.model_utils import evaluate
    from .pytorch_models.model_utils import save_model
    from .pytorch_models.model_utils import print_summary
    from .pytorch_models.model_utils import fit, predict, predict_losses
    from .pytorch_models.model_utils import get_model_architecture
    from .pytorch_models.model_utils import init_model
    from .pytorch_models.model_utils import calculate_loss
    from .pytorch_models.model_utils import count_params