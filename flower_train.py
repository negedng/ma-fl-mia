import flwr as fl
from flower_client import FlowerClient
from flwr.common.logger import log
from logging import ERROR, INFO

ROOT_PATH = utils.lookup_envroot()

global conf = {
    'unit_size':64,
    'num_clients':20,
    'models_path': ROOT_PATH+"/data/models/ma-fl-mia/federated/",
    'codes_path': ROOT_PATH+"/data/codes/ma-fl-mia/flower_train.py",
    'seed': 20,
    'rounds': 5,
    'epochs': 1
}
global X_split = None
global Y_split = None


def client_fn(cid: str) -> fl.client.Client:
    """Prepare flower client from ID (following flower documentation)"""
    client = FlowerClient(int(cid), conf)
    client.load_data(X_split[int(cid)], Y_split[int(cid)])
    client.init_model()
    return client


class SaveAndLogStrategy(fl.server.strategy.FedAvg):
    """Adding saving and logging to the strategy pipeline"""

    def __init__(self, conf, *args, **kwargs):
        self.conf = conf
        self.client_leaving_history = []
        self.aggregated_parameters = None
        self.best_loss = np.inf
        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        rnd,
        results,
        failures,
    ):
        """Aggregate model weights using weighted average and store best checkpoint"""
        aggregated_parameters_tuple = super().aggregate_fit(rnd, results, failures)
        self.aggregated_parameters, _ = aggregated_parameters_tuple

        return aggregated_parameters_tuple

    def aggregate_evaluate(
        self,
        rnd,
        results,
        failures,
    ):
        aggregated_result = super().aggregate_evaluate(rnd, results, failures)
        log(
            INFO,
            "Aggregated results: %s",
            aggregated_result,
        )
        if self.aggregated_parameters is not None and aggregated_result[0] < self.best_loss:
            self.best_loss = aggregated_result[0]
            log(INFO, "Saving model")
            aggregated_weights = fl.common.parameters_to_ndarrays(
                self.aggregated_parameters)
            model = models.get_model(self.conf.unit_size)
            model.compile(optimizer=models.get_optimizer(),
                          loss=models.get_loss())
            model.set_weights(aggregated_weights)
            model.save(os.path.join(conf.models_path,
                       conf.model_id, 'saved_model'))
        if rnd == self.conf.ROUNDS:
            # end of training calls
            pass
        return aggregated_result


if __name__ == "__main__":
    import tensorflow as tf 
    import tensorflow_datasets as tfds
    
    conf.model_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(conf.models_path, conf.model_id), mode=0o777)
    shutil.copy(conf.codes_path,
                os.path.join(conf.models_path, conf.model_id, "train.py"))
    
    train_ds, test_ds = tfds.load('cifar10', split=['train','test'], as_supervised=True)
    X_train, Y_train = utils.get_np_from_tfds(train_ds)
    X_split, Y_split = utils.split_data(X_train, Y_train, conf.num_clients, mode="clients", seed=conf.seed)

    # Create FedAvg strategy
    strategy = SaveAndLogStrategy(
        conf=conf,
        fraction_fit=1.0,  # Sample 10% of available clients for training
        fraction_evaluate=0.05,  # Sample 5% of available clients for evaluation
        min_fit_clients=1,  # Never sample less than 10 clients for training
        min_evaluate_clients=1,  # Never sample less than 5 clients for evaluation
        # Wait until at least 75 clients are available
        # min_available_clients=1,
    )

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=conf.num_clients,
        config=fl.server.ServerConfig(num_rounds=conf.rounds),
        strategy=strategy,
    )
    
    # Evaluate
    model = tf.keras.models.load_model(os.path.join(conf.models_path, conf.model_id, "saved_model"),
                                       custom_objects={})
    model.compile(optimizer=models.get_optimizer(),
                  loss=models.get_loss())

