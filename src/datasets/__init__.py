from src import TF_MODELS

if TF_MODELS:
    from .tensorflow_datasets.augmentation import aug_data
    from .tensorflow_datasets.data_preparation import get_ds_from_np
    from .tensorflow_datasets.data_preparation import get_np_from_ds
    from .tensorflow_datasets.data_preparation import preprocess_data
    from .tensorflow_datasets.data_preparation import load_data
else:
    from .pytorch_datasets.augmentation import aug_data
    from .pytorch_datasets.data_preparation import load_data
    from .pytorch_datasets.data_preparation import preprocess_data
    from .pytorch_datasets.data_preparation import get_np_from_ds
    from .pytorch_datasets.data_preparation import get_ds_from_np
