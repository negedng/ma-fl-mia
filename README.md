# ma-fl-mia

Run `flower_train.py` for federated training, `central_train.py` for centralized training. Configure hyperparameters in `config.json`.

Model-agnostic matrix mapping in `src/model_aggregation.py`

Client handler is in `src/flower_client.py`

Server to client message handling in `src/flower_strategy.py` and `src/model_aggregation.py`

TensorFlow models are obsolete, use PyTorch instead! Model aggregation is framework-independent, but latest versions are tested only on PyTorch.

# FEMNIST
Generating sample following original `leaf` code:

`./preprocess.sh -s niid --sf 0.2 -k 300 -t user --smplseed 0 --spltseed 0`

Don't forget to change `data_preparation.load_data()` if you're using a different sample generation.

Writers to client partitioning:
`[3,6,9,12,15,18,21,24,27]` and rest of the clients (`32` in this seed)

Example data partition: `[11267, 9562, 8603, 7318, 6530, 5275, 4357, 3184, 2162, 1001]`

# Config

 - `dataset`: for CIFAR10, CIFAR100, or FEMNIST dataset
 - `num_clients`: for number of clients to train with
 - `rounds`: to configurate number of federated rounds, `epochs` for the local training epochs.
 - `split_mode`: `homogen` for IID distribution, `balanced` for class-balanced, sample size inbalanced data, `dirichlet` for Dirichlet distribution
 - `ma_mode`: `no` for FedAvg, `heterofl` for HeteroFL, `fjord` for FjORD, `rm-cid` for our model-agnostic methods (including a second implementation of HeteroFL)
 - `unit_size`: to configure size of the models. The four complexity level tested: 8,16,32,64.
 - `scale_mode`: variations of Large client, small client ratios:
   - Experiments in paper: `0`-`9`integers to determine the number of big clients [0,`num_clients`)
   - [0.0,1.0) for scaling every client down `r=unit_size_client/unit_size_server` (similarly to original FDropout)
   - (-`num_clients`,0) for `unit_size_client=unit_size_server-r` reduced size by integer number of channels
 - `permutate_cuts`: `repeated` for Fixed and `incremental` for Reshampled in the paper taxonomy. `one-repeated` and `one-incremental` for one group, `group-repeated` and `group-incremental` for several groups, and `repeated` or `incremental` for unique.
 - `cut_type`: `random` for Random or `submatrix` for Submatrix.
 - `cut_layerwise`: for experiment with same name in Appendix

# Named variations
 - FDropout: `{"cut_type": "random", "permutate_cuts":"incremental"}`
 - MaPP-FL (M): `{"cut_type": "submatrix", "permutate_cuts":"group-repeated"}` or `{"cut_type": "submatrix", "permutate_cuts":"repeated"}`
 - MaPP-FL (R): `{"cut_type": "random", "permutate_cuts":"group-repeated"}`
 - HeteroFL: `{"cut_type": "submatrix", "permutate_cuts":"one-repeated"}`
 - HeteroFL (R): `{"cut_type": "random", "permutate_cuts":"one-repeated"}`

