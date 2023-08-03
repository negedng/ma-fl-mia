import numpy as np
import itertools
from src import utils
from src import IN_CHANNEL_DIM, OUT_CHANNEL_DIM
import pdb


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


def take_new(layer_matrix, take_index_list):
    z = layer_matrix
    for i, cut_list in enumerate(take_index_list):
        z = z.take(cut_list, axis=i)
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
        cut_idx(to_shape, M_shape, i, conf=conf, rand=rand) for i in range(len(M_shape))
    ]

    M_padded = expand_with_index(M, to_shape, idx)
    return M_padded


def get_idx(from_len, to_len, conf, rand, next_2d):
    if rand is None:
        rand = 0  #!TODO heterofl shortcut
    if rand % 4 == 0:
        return list(range(to_len))
    if rand % 4 == 2:
        if next_2d % 2 == 0:
            return list(range(to_len))
        else:
            return list(range(from_len-to_len, from_len, 1))
    if rand % 4 == 1:
        if next_2d % 2 == 0:
            return list(range(from_len-to_len, from_len, 1))
        else:
            return list(range(to_len))
    if rand % 4 == 3:
        return list(range(from_len-to_len, from_len, 1))




def cut_idx_new(w_from_shape, w_to_shape, conf={}, rand=None):
    def is_channel_in(dim, wmatrix_shape):
        max_dim = len(wmatrix_shape)
        if IN_CHANNEL_DIM < 0:
            return max_dim + IN_CHANNEL_DIM == dim
        return IN_CHANNEL_DIM == dim

    def is_channel_out(dim, max_dim):
        max_dim = len(w_from_shape)
        if OUT_CHANNEL_DIM < 0:
            return max_dim + OUT_CHANNEL_DIM == dim
        return OUT_CHANNEL_DIM == dim

    w_idx = []
    last_out_idx = []
    next_2d = 0
    for l_from_shape, l_to_shape in zip(w_from_shape, w_to_shape):
        changing_dims = [
            dim
            for dim in range(len(l_from_shape))
            if l_from_shape[dim] != l_to_shape[dim]
        ]
        if len(changing_dims) == 2:
            next_2d += 1

        l_idx = []
        this_out_idx = []
        for dim, _ in enumerate(l_from_shape):
            from_len = l_from_shape[dim]
            to_len = l_to_shape[dim]
            if from_len == to_len:
                l_idx.append(list(range(from_len)))
            else:
                if (
                    conf["cut_type"] == "layer_fixed_submatrix"
                    or conf["cut_type"] == "simple"
                ):
                    l_idx.append(get_idx(from_len, to_len, conf, rand, dim))
                elif conf["cut_type"] == "layer_same_as_input":
                    if is_channel_in(dim, l_from_shape):
                        l_idx.append(last_out_idx)
                    elif is_channel_out(dim, l_from_shape):
                        this_out_idx = get_idx(from_len, to_len, conf, rand, next_2d)
                        l_idx.append(this_out_idx)
                    else:
                        raise IndexError(
                            "Something is wrong with IN - OUT channel dimension, expected (..,IN,OUT) or (OUT,IN,..)"
                        )
                else:
                    raise NotImplementedError(
                        "cut_type not recognized", conf["cut_type"]
                    )
        w_idx.append(l_idx)
        last_out_idx = this_out_idx
    return w_idx


def crop_weights(w_from, w_to, conf={}, rand=None):
    """Alternate top-right and bottom left to have the same out as the next layer's in"""
    w_from_shape = [l.shape for l in w_from]
    w_to_shape = [l.shape for l in w_to]
    idx_ret = cut_idx_new(w_from_shape, w_to_shape, conf=conf, rand=rand)
    w_ret = []
    for l_from, idx_list in zip(w_from, idx_ret):
        w_ret.append(take_new(l_from, idx_list))
    return w_ret


def crop_weights_old(w_from, w_to, conf={}, rand=None):
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


def aggregate_rmcid_old(
    results,
    rands,
    total_model_shapes,
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
            print(max_ch, count_layer.shape)
            # print(idx)
            print(rands)
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
    # print(cids, rands)
    agg_layers = [
        aggregate_layer(l, num_examples_list, rands, total_model_shapes[i], conf=conf)
        for i, l in enumerate(zip(*weighted_weights))
    ]
    return agg_layers


def aggregate_rmcid(
    results,  # [(w,no)...]
    rands,  # [cid..]
    total_model_shapes,
    conf={},
):
    def aggregate_layer(layer_updates, num_examples_list, cut_idx_list, max_dim_layer):
        """Padding layers with 0 to max size, then average them"""
        # In tensorflow biases have their list items in the layer list
        # Get the layer's largest form
        layers_padded = []
        layers_mask = []
        layer_agg = np.zeros(max_dim_layer)
        count_layer = np.zeros(max_dim_layer)  # to average by num of models that size
        for l, num, cut_id in zip(layer_updates, num_examples_list, cut_idx_list):
            l_padded = expand_with_index(l, max_dim_layer, cut_id)
            ones_pad = np.ones(np.shape(l))
            ones_pad = expand_with_index(ones_pad, max_dim_layer, cut_id)
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
            print(max_dim_layer, count_layer.shape)
            # print(idx)
            print(rands)
            raise ValueError("Diving with 0")
        layer_agg = layer_agg / count_layer
        return layer_agg

    # Calculate the total number of examples used during training
    num_examples_list = [num_examples for _, num_examples in results]
    num_examples_total = sum(num_examples_list)
    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]  # [w...]

    # Get cut matching
    cut_idx_list = [
        cut_idx_new(total_model_shapes, [l.shape for l in w_client], conf, rand)
        for w_client, rand in zip(weighted_weights, rands)
    ]
    # pdb.set_trace()
    agg_layers = [
        aggregate_layer(
            layer_updates, num_examples_list, cut_idxs_by_layer, total_model_shapes[i]
        )
        for i, (layer_updates, cut_idxs_by_layer) in enumerate(
            zip(zip(*weighted_weights), zip(*cut_idx_list))
        )
    ]
    return agg_layers
