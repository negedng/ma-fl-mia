import numpy as np
from typing import List, Tuple
from flwr.common import NDArray, NDArrays


def take(a, new_shape):
    """Takes the top-left submatrix with given shape"""
    z = a
    for i, d in enumerate(new_shape):
        z = z.take(range(d), axis=i)
    return z


def crop_weights(w_from: List[NDArrays], w_to: List[NDArrays]) -> List[NDArrays]:
    """Crop top-left matrix of weights from first list of arrays to second's shape"""
    w_ret = []
    for l_from, l_to in zip(w_from, w_to):
        l_to = take(l_from, np.shape(l_to))
        w_ret.append(l_to)
    return w_ret


def aggregate_hetero(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average with different model sizes."""
    def aggregate_layer(layer_updates, num_examples_list):
        """Padding layers with 0 to max size, then average them"""
        # Get the layer's largest form
        max_ch = np.max([np.shape(l) for l in layer_updates], axis=0)
        layers_padded = []
        layers_mask = []
        layer_agg = np.zeros(max_ch)
        count_layer = np.zeros(max_ch) # to average by num of models that size
        for l, num in zip(layer_updates, num_examples_list):
            local_ch = np.shape(l)
            pad_shape = [(0, a) for a in (max_ch - local_ch)]
            l_padded = np.pad(l, pad_shape, constant_values = 0.0)
            ones_of_shape = np.ones(local_ch) * num
            ones_pad = np.pad(ones_of_shape, pad_shape, constant_values = 0.0)
            count_layer = np.add(count_layer, ones_pad)
            layer_agg = np.add(layer_agg, l_padded)
        layer_agg = layer_agg / count_layer
        return layer_agg

    # Calculate the total number of examples used during training
    num_examples_list = [num_examples for _, num_examples in results]
    num_examples_total = sum(num_examples_list)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]
    # Aggregate the layers
    agg_layers = [
        aggregate_layer(l,num_examples_list) for l in zip(*weighted_weights)
    ]
    return agg_layers
