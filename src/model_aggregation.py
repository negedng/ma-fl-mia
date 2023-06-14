import numpy as np
import itertools
from src import utils


def cut_idx(max_shape, this_shape, dim, conf, rand):
    if conf["cut_type"] == "rand":
        return cut_idx_rand(max_shape, this_shape, dim, rand)
    if conf["cut_type"] == "secure":
        return cut_idx_rand_secure_first(max_shape, this_shape, dim, rand)
    if conf["cut_type"] == "diagonal":
        return cut_idx_diagonal(max_shape, this_shape, dim, rand)
    if conf["cut_type"] == "simple":
        return cut_idx_simple(max_shape, this_shape, dim, rand)
    raise ValueError(f'Not recognized cut_type: {conf["cut_type"]}')


def cut_idx_rand(max_shape, this_shape, dim, rand):
    from_len = max_shape[dim]
    to_len = this_shape[dim]
    if from_len == to_len:
        return np.array(range(to_len))
    remove_len = from_len - to_len
    r = np.random.RandomState(seed=rand * (dim + 1)).randint(to_len)
    remove_idx = range(r, r + remove_len)
    keep_idx = [x for x in range(from_len) if x not in remove_idx]
    keep_idx = np.array(keep_idx)
    return keep_idx


def cut_idx_diagonal(max_shape, this_shape, dim, rand):
    from_len = max_shape[dim]
    to_len = this_shape[dim]
    if from_len == to_len:
        return np.array(range(to_len))
    remove_len = from_len - to_len
    r = np.random.RandomState(seed=rand).randint(to_len)
    remove_idx = range(r, r + remove_len)
    keep_idx = [x for x in range(from_len) if x not in remove_idx]
    keep_idx = np.array(keep_idx)
    return keep_idx


def cut_idx_rand_secure_first(max_shape, this_shape, dim, rand):
    """First tries to match each unit to one submatrix then random for the rest"""
    from_len = max_shape[dim]
    to_len = this_shape[dim]
    if from_len == to_len:
        return np.array(range(to_len))

    steps_per_dim = np.ceil(np.array(max_shape) / np.array(this_shape)).astype(int)
    min_subs = np.prod(steps_per_dim)
    cid = rand
    if cid < min_subs:
        cid_r = utils.generalized_positional_notation(cid, steps_per_dim)[dim]

        start = (cid_r * to_len) % from_len
        # no overlap
        end = min(start + to_len, from_len)
        start = end - to_len

        p = list(range(from_len))
        p = np.concatenate([p, p])
        keep_idx = p[start : start + to_len]
        return keep_idx

    remove_len = from_len - to_len
    r = np.random.RandomState(seed=rand * (dim + 1)).randint(to_len)
    remove_idx = range(r, r + remove_len)
    keep_idx = [x for x in range(from_len) if x not in remove_idx]
    keep_idx = np.array(keep_idx)
    return keep_idx


def cut_idx_simple(max_shape, this_shape, dim, rand):
    """for 1/2 size, 4 matrixes, selecting one"""
    from_len = max_shape[dim]
    to_len = this_shape[dim]
    if from_len == to_len:
        return np.array(range(to_len))

    steps_per_dim = np.ceil(np.array(max_shape) / np.array(this_shape)).astype(int)
    min_subs = np.prod(steps_per_dim)
    cid = rand
    cid = cid % min_subs
    cid_r = utils.generalized_positional_notation(cid, steps_per_dim)[dim]

    start = (cid_r * to_len) % from_len
    # no overlap
    end = min(start + to_len, from_len)
    start = end - to_len

    p = list(range(from_len))
    p = np.concatenate([p, p])
    keep_idx = p[start : start + to_len]
    return keep_idx


def take(a, new_shape, conf={}, rand=None):
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
                take_index_list = cut_idx(a_shape, new_shape, i, conf=conf, rand=rand)
            z = z.take(take_index_list, axis=i)
    return z


def expand_with_index(B, A_shape, idx):
    """Expand B to the shape of A_shape and put values to the rows defined by idx"""
    if np.all(np.array(B.shape) == np.array(A_shape)):
        return B
    for i, (x, y) in enumerate(zip(B.shape, idx)):
        assert x == len(y), f"{x}!={len(y)} at {i}"
    C = np.zeros(A_shape).flatten()
    B = B.flatten()
    for i, r in enumerate(itertools.product(*idx)):
        j = r[0]
        for k, l in zip(A_shape[1:], r[1:]):
            j = j * k + l
        C[j] = B[i]
    C = C.reshape(A_shape)
    return C


def expand_matrix_conf(M, to_shape, conf={}, rand=None):
    """Pad matrix with zeros to shape"""
    M_shape = np.shape(M)
    idx = [
        cut_idx(to_shape, M_shape, i, conf=conf, rand=rand)
        for i in range(len(M_shape))
    ]

    M_padded = expand_with_index(M, to_shape, idx)
    return M_padded


def crop_weights(w_from, w_to, conf={}, rand=None):
    """Crop top-left matrix of weights from first list of arrays to second's shape"""
    w_ret = []
    for l_from, l_to in zip(w_from, w_to):
        l_to = take(l_from, np.shape(l_to), conf=conf, rand=rand)
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
        count_layer = np.zeros(max_ch)  # to average by num of models that size
        for l, num in zip(layer_updates, num_examples_list):
            local_ch = np.shape(l)
            pad_shape = [(0, a) for a in (max_ch - local_ch)]
            l_padded = np.pad(l, pad_shape, constant_values=0.0)
            ones_of_shape = np.ones(local_ch) * num
            ones_pad = np.pad(ones_of_shape, pad_shape, constant_values=0.0)
            count_layer = np.add(count_layer, ones_pad)
            layer_agg = np.add(layer_agg, l_padded)
        if np.any(count_layer == 0.0):
            print(count_layer)
            raise ValueError("Diving with 0")
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
    agg_layers = [aggregate_layer(l, num_examples_list) for l in zip(*weighted_weights)]
    return agg_layers


def aggregate_rmcid(
    results,
    cids,
    total_model_shapes,
    server_round,
    total_clients,
    permutate=True,
    conf={},
):
    """Expand client model weights missing 1 row&col and aggregate"""
    """Compute weighted average with different model sizes."""

    def aggregate_layer(layer_updates, num_examples_list, rands, max_ch, conf={}):
        """Padding layers with 0 to max size, then average them"""
        # In tensorflow biases have their list items in the layer list
        # Get the layer's largest form
        layers_padded = []
        layers_mask = []
        layer_agg = np.zeros(max_ch)
        count_layer = np.zeros(max_ch)  # to average by num of models that size
        for l, num, rand in zip(layer_updates, num_examples_list, rands):
            l_padded = expand_matrix_conf(l, max_ch, conf=conf, rand=rand)
            ones_pad = np.ones(np.shape(l))
            ones_pad = expand_matrix_conf(ones_pad, max_ch, conf=conf, rand=rand)
            ones_pad = ones_pad * num
            count_layer = np.add(count_layer, ones_pad)
            layer_agg = np.add(layer_agg, l_padded)

        if np.any(count_layer == 0.0):
            print(count_layer)
            print(ones_pad[0, 0, :, 0])
            print(ones_pad[0, 0, 0, :])
            print(count_layer[0, 0, 0, :])

            for l in layer_updates:
                print(np.shape(l))
            print(max_ch, local_ch)
            print(idx)
            print(cids)
            raise ValueError("Diving with 0")
        layer_agg = layer_agg / count_layer
        return layer_agg

    # Calculate the total number of examples used during training
    num_examples_list = [num_examples for _, num_examples in results]
    num_examples_total = sum(num_examples_list)

    rands = utils.get_random_permutation_for_all(cids, server_round, len(cids), not(permutate))
    rands = [rands[cid] for cid in cids]

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]
    # Aggregate the layers
    print(cids, rands)
    agg_layers = [
        aggregate_layer(l, num_examples_list, rands, total_model_shapes[i], conf=conf)
        for i, l in enumerate(zip(*weighted_weights))
    ]
    return agg_layers
