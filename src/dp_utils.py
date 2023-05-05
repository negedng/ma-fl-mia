import math
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy


def get_target_delta(data_size: int) -> float:
    den = 10 ** int(math.log10(data_size))
    return 1 / den


def estimate_epsilon_per_round(conf):
    local_dataset_size = conf['len_total_data']/conf['num_clients']
    delta = get_target_delta(local_dataset_size)
    sample_rate = conf['batch_size']/local_dataset_size
    noise_multiplier= conf['dp_noise']
    max_grad_norm = conf['dp_clipnorm']
    epochs = conf['epochs']
    
    compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=local_dataset_size,
                                              batch_size=conf['batch_size'],
                                              noise_multiplier=noise_multiplier,
                                              epochs=epochs,
                                              delta=delta)

