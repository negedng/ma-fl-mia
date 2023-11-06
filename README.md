# ma-fl-mia

Model Agnostic matrix mapping in `src/ma_utils.py`

Client handler is in `src/flower_client.py`

Server to client message handling in `src/flower_strategy.py` and `src/ma_utils.py`

Example of attack on client in `notebooks/AttacksOnClients.ipynb`

# FEMNIST
Generating sample following original `leaf` code:

`./preprocess.sh -s niid --sf 0.2 -k 300 -t user --smplseed 0 --spltseed 0`

Don't forget to change `data_preparation.load_data()` if you're using a different sample generation.

Writers to client partitioning:
`[3,6,9,12,15,18,21,24,27]` and rest of the clients (`32` in this seed)

Example data partition: `[11267, 9562, 8603, 7318, 6530, 5275, 4357, 3184, 2162, 1001]`

