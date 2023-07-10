import numpy as np


def attack_test():
    from src import attacks

    attack_knowledge_train_losses = [0.0,1.0,0.0,1.0,0.0,1.0]
    print(f"Simulated losses: {attack_knowledge_train_losses}")
    avg_threshold = attacks.yeom_standard_threshold(attack_knowledge_train_losses)
    print(f"Average threshold (0.5): {avg_threshold}")
    assert avg_threshold==0.5
    
    mia_test_losses = np.array([1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0])
    mia_test_true = np.array([0,0,0,0,0,1,1,1,1,1])
    mia_test_predicted = attacks.yeom_mi_attack(mia_test_losses, avg_threshold)
    attack_adv = attacks.calculate_advantage(mia_test_true, mia_test_predicted)
    print(f"Simulated attack, everyone right, expected adv 100.0: {attack_adv}") 
    assert attack_adv==100

    mia_test_true = np.array([1,1,1,1,1,0,0,0,0,0])
    mia_test_predicted = attacks.yeom_mi_attack(mia_test_losses, avg_threshold)
    attack_adv = attacks.calculate_advantage(mia_test_true, mia_test_predicted)
    print(f"Simulated attack, everyone wrong, expected adv -100.0: {attack_adv}") 
    assert attack_adv==-100    

    mia_test_true = np.array([1,1,1,1,1,1,1,1,1,1])
    mia_test_predicted = attacks.yeom_mi_attack(mia_test_losses, avg_threshold)
    attack_adv = attacks.calculate_advantage(mia_test_true, mia_test_predicted)
    print(f"Simulated attack, half true, expected adv 0.0: {attack_adv}") 
    assert attack_adv==0    


def split_test():
    print("Vanilla FL aggregation test:")
    client_weights = [np.ones((4,4)), np.ones((8,3,3,3)), np.ones((8))]
    client_weights_a = [l*0 for l in client_weights]
    client_weights_b = [l*2 for l in client_weights]
    n_clients = 4
    fed_weights = [client_weights_a] * (n_clients//2) + [client_weights_b] * (n_clients//2)
    num_samples = [1] * n_clients
    results = [(c_w, c_n) for c_w, c_n in zip(fed_weights, num_samples)]
    print(f"{n_clients} clients with data shape:")
    print([l.shape for l in client_weights])
    
    from flwr.server.strategy.aggregate import aggregate
    agg_layers = aggregate(results)
    print("Aggregated layers to shape:")
    print([l.shape for l in agg_layers])
    for l1,l2 in zip(client_weights, agg_layers):
        assert l1.shape==l2.shape
    print(client_weights_a[0]," \n+\n",client_weights_b[0],"\n avg:\n", agg_layers[0])
    assert np.all(agg_layers[0])==1.0
    print("HeteroFL aggregation test:")
    server_weights = [np.arange(1,17).reshape(4,4), np.ones((8,3,3,3)), np.ones((8))]
    print(server_weights[0])
    print("Server shape:       ", [l.shape for l in server_weights])
    from src import model_aggregation
    small_client_weights_temp = [np.zeros((2,2)), np.ones((4,3,3,3)), np.ones((4))]
    print("Small client shape: ", [l.shape for l in small_client_weights_temp])
    small_client_weights = model_aggregation.crop_weights(server_weights, small_client_weights_temp)
    for l1,l2 in zip(small_client_weights, small_client_weights_temp):
        assert l1.shape==l2.shape
    print(small_client_weights[0])
    assert np.all(server_weights[0][:2,:2]==small_client_weights[0])
    small_client_weights = [l*2 for l in small_client_weights]
    large_client_weights = [l*0 for l in server_weights]
    results = [(c_w, c_n) for c_w, c_n in zip([small_client_weights, large_client_weights], [1,1])]
    agg_layers = model_aggregation.aggregate_hetero(results)
    print(agg_layers[0])
    assert np.all(server_weights[0][:2,:2]==agg_layers[0][:2,:2])
    print("RM-CID test")
    server_weights = [np.arange(1,17).reshape(4,4), np.arange(0,(8*16*3*3)).reshape((8,16,3,3)), np.ones((8))]
    print("Server shape:       ", [l.shape for l in server_weights])
    client_weights_temp = [np.zeros((2,2)), np.zeros((4,8,3,3)), np.ones((4))]
    print(server_weights[0])
    print("Small client shape: ", [l.shape for l in client_weights_temp])
    client_0 = model_aggregation.crop_weights(server_weights, client_weights_temp,{"cut_type":"simple"},0)
    client_1 = model_aggregation.crop_weights(server_weights, client_weights_temp,{"cut_type":"simple"},1)
    client_2 = model_aggregation.crop_weights(server_weights, client_weights_temp,{"cut_type":"simple"},2)
    client_3 = model_aggregation.crop_weights(server_weights, client_weights_temp,{"cut_type":"simple"},3)
    print(client_0[0],"--\n",client_1[0],"--\n",client_2[0],"--\n",client_3[0])
    results = [(c_w, c_n) for c_w, c_n in zip([client_0,client_1,client_2,client_3], [1,1,1,1])]
    # Permutate is the opposite of what it means because that refers to the takeout part
    server_round = 0
    agg_layers = model_aggregation.aggregate_rmcid(results,[0,1,2,3],[l.shape for l in server_weights],
                                                   server_round,4,conf={"cut_type":"simple"}, permutate=True)
    print(agg_layers[0])
    for l1,l2 in zip(server_weights, agg_layers):
        assert np.all(l1==l2)
    server_round = 1
    agg_layers = model_aggregation.aggregate_rmcid(results,[0,1,2,3],[l.shape for l in server_weights],
                                                   server_round,4,conf={"cut_type":"simple"}, permutate=True)
    for l1,l2 in zip(server_weights, agg_layers):
        assert np.all(l1==l2)
    print("Putting back with permutation (random permutation of the submatrix expected):")
    server_round = 0
    agg_layers = model_aggregation.aggregate_rmcid(results,[0,1,2,3],[l.shape for l in server_weights],
                                                   server_round,4,conf={"cut_type":"simple"}, permutate=False)
    print(agg_layers[0])
    server_round = 1
    agg_layers = model_aggregation.aggregate_rmcid(results,[0,1,2,3],[l.shape for l in server_weights],
                                                   server_round,4,conf={"cut_type":"simple"}, permutate=False)
    print(agg_layers[0])
    print("Clients cut randomly, server aggregates to the same place (client cuts not in order, but aggregation same as begining):")
    from src import utils
    server_round = 3
    rands = utils.get_random_permutation_for_all([0,1,3,2],
                                                server_round, 
                                                4, 
                                                True)
    print(rands)
    client_0 = model_aggregation.crop_weights(server_weights, client_weights_temp,{"cut_type":"simple"},rands[0])
    client_1 = model_aggregation.crop_weights(server_weights, client_weights_temp,{"cut_type":"simple"},rands[1])
    client_2 = model_aggregation.crop_weights(server_weights, client_weights_temp,{"cut_type":"simple"},rands[2])
    client_3 = model_aggregation.crop_weights(server_weights, client_weights_temp,{"cut_type":"simple"},rands[3])
    print(client_0[0],"--\n",client_1[0],"--\n",client_2[0],"--\n",client_3[0])
    results = [(c_w, c_n) for c_w, c_n in zip([client_0,client_1,client_2,client_3], [1,1,1,1])]
    agg_layers = model_aggregation.aggregate_rmcid(results,[0,1,2,3],[l.shape for l in server_weights],
                                                   server_round,4,conf={"cut_type":"simple"}, permutate=False)
    print(agg_layers[0])
    for l1,l2 in zip(server_weights, agg_layers):
        assert np.all(l1==l2)
    print("10 clients, putting back to the same location (expected server weights):")                          
    server_round = 111
    n_clients = 10
    rands = utils.get_random_permutation_for_all(list(range(n_clients)),
                                        server_round, 
                                        n_clients, 
                                        True)
    print(rands)
    clients = [model_aggregation.crop_weights(server_weights, client_weights_temp,{"cut_type":"simple"},rands[i])
        for i in range(n_clients)
    ]
    results = [(c_w, c_n) for c_w, c_n in zip(clients, [1]*n_clients)]
    agg_layers = model_aggregation.aggregate_rmcid(results,list(range(n_clients)),[l.shape for l in server_weights],
                                                   server_round,n_clients,conf={"cut_type":"simple"}, permutate=False)
    print(agg_layers[0])
    for l1,l2 in zip(server_weights, agg_layers):
        assert np.all(l1==l2)
    
    n_clients = 10000
    print(f"{n_clients} clients, putting back to the other location, expected 4 submatrix converging to the average of the elements")
    print("Expected:")
    e = [(client_0[i]+client_1[i]+client_2[i]+client_3[i])/4 for i in range(len(client_0))]
    e = [(c_w, c_n) for c_w, c_n in zip([e,e,e,e], [1,1,1,1])]
    e = model_aggregation.aggregate_rmcid(e,[0,1,2,3],[l.shape for l in server_weights],
                                          server_round,4,conf={"cut_type":"simple"}, permutate=False)
    print(e[0])                          
    server_round = 111
    rands = utils.get_random_permutation_for_all(list(range(n_clients)),
                                        server_round, 
                                        n_clients, 
                                        True)
    # print(rands)
    clients = [model_aggregation.crop_weights(server_weights, client_weights_temp,{"cut_type":"simple"},rands[i])
        for i in range(n_clients)
    ]
    results = [(c_w, c_n) for c_w, c_n in zip(clients, [1]*n_clients)]
    agg_layers = model_aggregation.aggregate_rmcid(results,list(range(n_clients)),[l.shape for l in server_weights],
                                                   server_round,n_clients,conf={"cut_type":"simple"}, permutate=True)
    print(agg_layers[0])
    for l1,l2 in zip(e, agg_layers):
        print(np.max(np.abs(l1-l2)))
    print("Several rounds with 10 clients")
    n_clients = 10
    agg_layers = server_weights
    for server_round in range(150):
        rands = utils.get_random_permutation_for_all(list(range(n_clients)),
                                            server_round, 
                                            n_clients, 
                                            True)
        clients = [model_aggregation.crop_weights(agg_layers, client_weights_temp,{"cut_type":"simple"},rands[i])
            for i in range(n_clients)
        ]
        results = [(c_w, c_n) for c_w, c_n in zip(clients, [1]*n_clients)]
        agg_layers = model_aggregation.aggregate_rmcid(results,list(range(n_clients)),[l.shape for l in server_weights],
                                                    server_round,n_clients,conf={"cut_type":"simple"}, permutate=True)  
    print(agg_layers[0])
    for l1,l2 in zip(e, agg_layers):
        print(np.max(np.abs(l1-l2)))

if __name__ == "__main__":
    print("Attack test:")
    attack_test()
    print("Test splits:")
    split_test()
    print("Tests OK")
    