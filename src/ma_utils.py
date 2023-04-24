import numpy as np
import itertools

def cut_for_rand(from_shape, to_shape, rand):
    remove_shape = from_shape - to_shape

    r1 = np.random.RandomState(seed=rand).randint(from_shape)
    last_idx = min(r1+remove_shape,from_shape)
    first_idx = last_idx-remove_shape
    remove_idx = range(first_idx,last_idx)
    keep_idx = [x for x in range(from_shape) if x not in remove_idx]
    return keep_idx


def take(a, new_shape, rand=None):
    """Takes the top-left submatrix with given shape, 
    if rand is not None, takes random submatrix defined by this int seed"""
    a_shape = np.shape(a)
    z = a
    for i, d in enumerate(new_shape):
        z_shape = np.shape(z)
        if z_shape[i] != d:
            if rand is None:
                take_index_list = range(d)
            else:
                take_index_list = cut_for_rand(a_shape[i], d, rand=rand)
            z = z.take(take_index_list, axis=i) 
    return z


def expand(B, A_shape, idx):
    """Expand B to the shape of A_shape and put values to the rows defined by idx"""
    if np.all(np.array(B.shape)==np.array(A_shape)):
        return B
    for i, (x,y) in enumerate(zip(B.shape, idx)):
        assert x==len(y), f'{x}!={len(y)} at {i}'
    C = np.zeros(A_shape).flatten()
    B = B.flatten()
    for i, r in enumerate(itertools.product(*idx)):
        j = r[0]
        for k,l in zip(A_shape[1:], r[1:]):
            j = j*k+l
        C[j] = B[i]
    C = C.reshape(A_shape)
    return C


def crop_weights(w_from, w_to, rand=None):
    """Crop top-left matrix of weights from first list of arrays to second's shape"""
    w_ret = []
    for l_from, l_to in zip(w_from, w_to):
        l_to = take(l_from, np.shape(l_to), rand=rand)
        w_ret.append(l_to)
    return w_ret


def aggregate_hetero(results):
    """Compute weighted average with different model sizes."""
    def aggregate_layer(layer_updates, num_examples_list):
        """Padding layers with 0 to max size, then average them"""
        # In tensorflow biases have their list items in the layer list
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


def aggregate_rmcid(results, cids, total_model_shapes):
    """Expand client model weights missing 1 row&col and aggregate"""
    """Compute weighted average with different model sizes."""
    def aggregate_layer(layer_updates, num_examples_list, cids, max_ch):
        """Padding layers with 0 to max size, then average them"""
        # In tensorflow biases have their list items in the layer list
        # Get the layer's largest form
        layers_padded = []
        layers_mask = []
        layer_agg = np.zeros(max_ch)
        count_layer = np.zeros(max_ch) # to average by num of models that size
        for l, num, cid in zip(layer_updates, num_examples_list, cids):
            local_ch = np.shape(l)
            idx = [
                cut_for_rand(max_ch[i], local_ch[i], rand=cid) if local_ch[i]<max_ch[i] else list(range(local_ch[i])) for i in range(len(local_ch)) 
            ]
            l_padded = expand(l, max_ch, idx)
            ones_pad = np.ones(np.shape(l))
            ones_pad = expand(ones_pad, max_ch, idx)
            ones_pad = ones_pad * num
            count_layer = np.add(count_layer, ones_pad)
            layer_agg = np.add(layer_agg, l_padded)
        
        #print(count_layer)
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
        aggregate_layer(l,num_examples_list, cids, total_model_shapes[i]) for i,l in enumerate(zip(*weighted_weights))
    ]
    return agg_layers
