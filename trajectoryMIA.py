
import os
import json
import copy
import numpy as np

from src.datasets import data_allocation
from src import datasets

from src import utils, models, attacks, metrics
from src.models import model_utils
from exp import setups
from src import WANDB_EXISTS
LIRA_ATTACK = True
TRAJECTORY_ATTACK = True
YEOM_ATTACK = True

import torch


from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union
from sklearn import metrics
import torch.nn.functional as F

class MLP_BLACKBOX(torch.nn.Module):
    def __init__(self, dim_in):
        super(MLP_BLACKBOX, self).__init__()
        self.dim_in = dim_in
        self.fc1 = torch.nn.Linear(self.dim_in, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 32)
        self.fc4 = torch.nn.Linear(32, 2)

    def forward(self, x):
        x = x.view(-1, self.dim_in)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x


def check_and_transform_label_format(
    labels: np.ndarray, nb_classes: Optional[int]= None, return_one_hot: bool = True
) -> np.ndarray:
    """
    Check label format and transform to one-hot-encoded labels if necessary

    :param labels: An array of integer labels of shape `(nb_samples,)`, `(nb_samples, 1)` or `(nb_samples, nb_classes)`.
    :param nb_classes: The number of classes.
    :param return_one_hot: True if returning one-hot encoded labels, False if returning index labels.
    :return: Labels with shape `(nb_samples, nb_classes)` (one-hot) or `(nb_samples,)` (index).
    """
    if labels is not None:
        if len(labels.shape) == 2 and labels.shape[1] > 1:
            if not return_one_hot:
                labels = np.argmax(labels, axis=1)
        elif len(labels.shape) == 2 and labels.shape[1] == 1 and nb_classes is not None and nb_classes > 2:
            labels = np.squeeze(labels)
            if return_one_hot:
                labels = to_categorical(labels, nb_classes)
        elif len(labels.shape) == 2 and labels.shape[1] == 1 and nb_classes is not None and nb_classes == 2:
            pass
        elif len(labels.shape) == 1:
            if return_one_hot:
                if nb_classes == 2:
                    labels = np.expand_dims(labels, axis=1)
                else:
                    labels = to_categorical(labels, nb_classes)
        else:
            raise ValueError(
                "Shape of labels not recognised."
                "Please provide labels in shape (nb_samples,) or (nb_samples, nb_classes)"
            )

    return labels


def to_categorical(labels: Union[np.ndarray, List[float]], nb_classes: Optional[int] = None) -> np.ndarray:
    """
    Convert an array of labels to binary class matrix.

    :param labels: An array of integer labels of shape `(nb_samples,)`.
    :param nb_classes: The number of classes (possible labels).
    :return: A binary matrix representation of `y` in the shape `(nb_samples, nb_classes)`.
    """
    labels = np.array(labels, dtype=np.int32)
    if nb_classes is None:
        nb_classes = np.max(labels) + 1
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1

    return categorical


def train_mia_attack_model(model, attack_train_loader, optimizer, loss_fn, device):
    model.to(device)
    model.train()
    train_loss = 0
    correct = 0
    
    for batch_idx, (model_loss_ori, model_trajectory, member_status) in enumerate(attack_train_loader):
        input = torch.cat((model_trajectory, model_loss_ori.unsqueeze(1)),1) 
        input = input.to(device)
        output = model(input)
        member_status = member_status.to(device)
        loss = loss_fn(output, member_status)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   
        train_loss += loss.item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(member_status.view_as(pred)).sum().item()

    train_loss /= len(attack_train_loader.dataset)
    accuracy = 100. * correct / len(attack_train_loader.dataset)
    return train_loss, accuracy/100.


def test_mia_attack_model(model, attack_test_loader, loss_fn, max_auc, max_acc, device):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    auc_ground_truth = None
    auc_pred = None
    with torch.no_grad():
        for batch_idx, (model_loss_ori, model_trajectory, member_status) in enumerate(attack_test_loader):

            input = torch.cat((model_trajectory, model_loss_ori.unsqueeze(1)),1) 
            input = input.to(device)
            output = model(input)
            member_status = member_status.to(device)
            test_loss += loss_fn(output, member_status).item()
            pred0, pred1 = output.max(1, keepdim=True)
            correct += pred1.eq(member_status.view_as(pred1)).sum().item()
            auc_pred_current = output[:, -1]
            auc_ground_truth = member_status.cpu().numpy() if batch_idx == 0 else np.concatenate((auc_ground_truth, member_status.cpu().numpy()), axis=0)
            auc_pred = auc_pred_current.cpu().numpy() if batch_idx == 0 else np.concatenate((auc_pred, auc_pred_current.cpu().numpy()), axis=0)

    test_loss /= len(attack_test_loader.dataset)
    accuracy = 100. * correct / len(attack_test_loader.dataset)

    fpr, tpr, thresholds = metrics.roc_curve(auc_ground_truth, auc_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    if auc > max_auc:
        max_auc = auc
        save_data = {
            'fpr': fpr,
            'tpr': tpr
        }
        np.save(f'trajectory_auc', save_data)
    if accuracy > max_acc:
        max_acc = accuracy

    return test_loss, accuracy/100., auc, max_auc, max_acc, tpr, fpr


def attack_fit(model, attack_train_loader, optimizer, loss_fn, device, epochs=100, attack_test_loader=None):
    max_auc = 0
    max_acc = 0
    for epoch in range(epochs):
        loss, acc = train_mia_attack_model(model, attack_train_loader, optimizer, loss_fn, device)
        if attack_test_loader is not None:
            val_loss, val_prec1, val_auc, max_auc, max_acc, _, _ = test_mia_attack_model(model, attack_test_loader, loss_fn, max_auc, max_acc, device)
            print(f'epoch: {epoch}, loss:{loss:.5f}, acc:{acc:.4f}, val_loss:{val_loss:.5f}, val_acc:{val_prec1:.4f}, val_auc:{val_auc:.2f}')
        else:
            print(f'epoch: {epoch}, loss:{loss}, acc:{acc}')


if __name__ == "__main__":
    import argparse

    # Instantiate the parser
    parser = argparse.ArgumentParser(
        description="Attack model with TrajectoryMIA attack"
    )
    parser.add_argument(
        "model", type=str, help="model id [yymmdd-hhmmss,...]"
    )
    parser.add_argument(
        "--cid",
        type=int,
        help="client id to attack",
        default=-1,
    )
    parser.add_argument(
        "--root",
        type=str,
        help="root path",
        default='/data/models/ma-fl-mia/federated/',
    )
    parser.add_argument(
        "--depochs",
        type=int,
        help="Distill epochs",
        default=25,
    )
    parser.add_argument(
        "--swnum",
        type=int,
        help="shadow models",
        default=16,
    )
    parser.add_argument(
        "--swepochs",
        type=int,
        help="shadow epochs",
        default=100,
    )
    args = parser.parse_args()
    shadow_epochs = args.swepochs
    if not (args.model[0]=='[' or args.model[0]=='{'):
        model_ids = [args.model]
    else:
        model_ids = json.loads(args.model)

    model_id = model_ids[0]
    config_path = os.path.join(args.root, model_id, 'config.json')
    with open(config_path) as f:
        conf = json.load(f)

    train_ds, val_ds, _ = datasets.load_data(conf=conf)

    if conf["dataset"]=="FEMNIST":
        X_split, Y_split = data_allocation.split_femnist(train_ds,
                                                        conf["num_clients"],
                                                        seed=conf["seed"])
        val_ds = datasets.get_np_from_femnist(val_ds)
        val_ds = datasets.get_ds_from_np(val_ds)
        print([len(y) for y in Y_split])
    else:
        X_train, Y_train = datasets.get_np_from_ds(train_ds)
        conf["len_total_data"] = len(X_train)
        X_split, Y_split = data_allocation.split_data(
            X_train,
            Y_train,
            conf["num_clients"],
            split_mode=conf["split_mode"],
            mode="clients",
            distribution_seed=conf["seed"],
            shuffle_seed=conf["data_shuffle_seed"],
            dirichlet_alpha=conf["dirichlet_alpha"],
        )

    if args.cid == -1:
        cids = list(np.arange(conf["num_clients"]))
    else:
        cids = [args.cid]

    all_results = {}
    print(cids)
    for client_id in cids:
        print(f"Data split for {client_id} CID")    

        X_one_left_out = X_split[0:client_id] + X_split[client_id+1:]
        Y_one_left_out = Y_split[0:client_id] + Y_split[client_id+1:]
        X_one_left_out = np.concatenate(X_one_left_out)
        Y_one_left_out = np.concatenate(Y_one_left_out)

        balanced_sample_size = min(len(X_split[client_id]),10000,conf['n_attack_sample'])

        cl_ds = datasets.get_ds_from_np((X_split[client_id][:balanced_sample_size], Y_split[client_id][:balanced_sample_size]))
        cl_ds = datasets.preprocess_data(cl_ds, conf=conf, shuffle=False)
        X_test, Y_test = datasets.get_np_from_ds(val_ds)
        test_ds = datasets.get_ds_from_np((X_test[:balanced_sample_size], Y_test[:balanced_sample_size]))
        test_ds = datasets.preprocess_data(test_ds, conf=conf, shuffle=False)
        
        print("Shadow model training")
        if LIRA_ATTACK:
            sw_num = args.swnum
            print(f"LiRA offline with {sw_num} models")
            from scipy.special import logit

            use_softmax = True
            use_logit = True

            pout_list = None
            pin_list = None

            for i in range(sw_num):
                print(i)
                shadow_idx = np.random.RandomState(seed=i).permutation(len(Y_one_left_out))[:10000]
                X_sw = X_one_left_out[shadow_idx]
                Y_sw = Y_one_left_out[shadow_idx]
                sw_ds = datasets.get_ds_from_np((X_sw, Y_sw))
                if conf["aug"]:
                    sw_ds = datasets.aug_data(sw_ds, conf=conf)
                sw_ds = datasets.preprocess_data(sw_ds, conf=conf, shuffle=True)

                swnet = model_utils.init_model(conf["unit_size"], conf=conf, static_bn=False,)
                sw_conf = copy.deepcopy(conf)
                sw_conf['epochs'] = shadow_epochs
                models.fit(swnet, sw_ds, conf=sw_conf, verbose=0)
                
                pout, lout = models.predict_ds(swnet, test_ds, apply_softmax=use_softmax, conf=conf)
                pin, lin = models.predict_ds(swnet, cl_ds, apply_softmax=use_softmax, conf=conf)
                
                pout = np.max(pout, -1)
                pin = np.max(pin, -1)
                if use_logit:

                    pout = np.clip(pout, 1e-4, 1-1e-4)
                    pin = np.clip(pin, 1e-4, 1-1e-4)
                    pout = logit(pout)
                    pin = logit(pin)
                
                if pout_list is None:
                    pout_list = pout
                    pin_list = pin
                else:
                    pout_list = np.vstack((pout_list,pout))
                    pin_list = np.vstack((pin_list, pin))
            
            pout_list = pout_list.transpose()
            pin_list = pin_list.transpose()
            pout_mean = np.mean(pout_list,1)
            pout_std = np.std(pout_list,1)
            pin_mean = np.mean(pin_list,1)
            pin_std = np.std(pin_list,1)
            print(pin_mean.shape, np.average(pin_mean), pout_mean.shape, np.average(pout_mean))

            # Clean up some space
            pout_list = None
            pin_list = None

        loss_function = models.get_loss(conf)
        if TRAJECTORY_ATTACK:
            print("TrajectoryMIA")
            swnet = model_utils.init_model(conf["unit_size"], conf=conf, static_bn=False,)
            models.print_summary(swnet)

            shuffled_indices = np.arange(len(X_one_left_out))
            np.random.default_rng(seed=conf['seed']).shuffle(shuffled_indices)

            X_sw = X_one_left_out[:(len(X_one_left_out)//4)]
            X_nsw = X_one_left_out[(len(X_one_left_out)//4):2*(len(X_one_left_out)//4)]
            X_dl = X_one_left_out[2*(len(X_one_left_out)//4):]

            Y_sw = Y_one_left_out[:(len(X_one_left_out)//4)]
            Y_nsw = Y_one_left_out[(len(X_one_left_out)//4):2*(len(X_one_left_out)//4)]
            Y_dl = Y_one_left_out[2*(len(X_one_left_out)//4):]

            print(f'Train: {len(X_split[client_id])}, Sw: {len(X_sw)}, NSw: {len(X_nsw)}, Distill: {len(X_dl)}')

            
            sw_ds = datasets.get_ds_from_np((X_sw, Y_sw))
            if conf["aug"]:
                sw_ds = datasets.aug_data(sw_ds, conf=conf)
            sw_ds = datasets.preprocess_data(sw_ds, conf=conf, shuffle=True)

            sw_conf = copy.deepcopy(conf)
            sw_conf['epochs'] = shadow_epochs
            models.fit(swnet, sw_ds, conf=sw_conf, verbose=1)

            attack_data = {}

            sw_ds = datasets.get_ds_from_np((X_sw, Y_sw))
            sw_ds = datasets.preprocess_data(sw_ds, conf=conf, shuffle=False)
            nsw_ds = datasets.get_ds_from_np((X_nsw, Y_nsw))
            nsw_ds = datasets.preprocess_data(nsw_ds, conf=conf, shuffle=False)

            in_losses = models.get_losses(swnet, sw_ds, loss_function, conf=conf)
            out_losses = models.get_losses(swnet, nsw_ds, loss_function, conf=conf)
            attack_data['model_loss_ori'] = np.concatenate((in_losses, out_losses))
            attack_data['member_status'] = np.array([1]*len(in_losses) + [0]*len(out_losses))

            print("Distill shadow model")
            dl_ds = datasets.get_ds_from_np((X_dl, Y_dl))
            if conf["aug"]:
                dl_ds = datasets.aug_data(dl_ds, conf=conf)
            dl_ds = datasets.preprocess_data(dl_ds, conf=conf, shuffle=True)

            distill_epochs = args.depochs

            distill_net = model_utils.init_model(conf["unit_size"], conf=conf, static_bn=False,)
            loss_traj_in = None
            loss_traj_out = None


            for i in range(distill_epochs):
                print(i)
                models.fit(distill_net, dl_ds, conf=conf, verbose=0, distill_target_model=swnet)
                
                loss_in = models.get_losses(distill_net, sw_ds, loss_function, conf=conf)
                loss_out = models.get_losses(distill_net, nsw_ds, loss_function, conf=conf)

                if loss_traj_in is None:
                    loss_traj_in = loss_in
                    loss_traj_out = loss_out
                else:
                    loss_traj_in = np.vstack((loss_traj_in, loss_in))
                    loss_traj_out = np.vstack((loss_traj_out, loss_out))
            attack_data['model_trajectory'] = np.concatenate((loss_traj_in.transpose(),loss_traj_out.transpose()))
        

        for model_id in model_ids:
            print(f"Distill target model: {model_id}")
            model_path = os.path.join(args.root, model_id, 'clients',str(client_id),'saved_model_post_'+str(conf["rounds"]))
            config_path = os.path.join(args.root, model_id, 'config.json')
            with open(config_path) as f:
                client_conf = json.load(f)
            assert conf['seed']==client_conf['seed']
            assert conf['data_shuffle_seed']==client_conf['data_shuffle_seed']

            unit_size = utils.calculate_unit_size(client_id, client_conf, len(X_split[client_id]))


            client_model = model_utils.init_model(
                        unit_size,
                        conf=client_conf,
                        model_path=model_path,
                        static_bn=True,
                    )
            client_model.to(models.get_device(conf))

            desired_fpr = [0.001, 0.005, 0.01, 0.03]
            if TRAJECTORY_ATTACK:
                distill_cnet = model_utils.init_model(conf["unit_size"], conf=conf, static_bn=False,)
                loss_traj_in = None
                loss_traj_out = None


                for i in range(distill_epochs):
                    print(i)
                    models.fit(distill_cnet, dl_ds, conf=conf, verbose=0, distill_target_model=client_model)
                    
                    loss_in = models.get_losses(distill_cnet, cl_ds, loss_function, conf=conf)
                    loss_out = models.get_losses(distill_cnet, test_ds, loss_function, conf=conf)

                    if loss_traj_in is None:
                        loss_traj_in = loss_in
                        loss_traj_out = loss_out
                    else:
                        loss_traj_in = np.vstack((loss_traj_in, loss_in))
                        loss_traj_out = np.vstack((loss_traj_out, loss_out))

                attack_test = {}
                in_losses = models.get_losses(client_model, cl_ds, loss_function, conf=conf)
                out_losses = models.get_losses(client_model, test_ds, loss_function, conf=conf)
                attack_test['model_loss_ori'] = np.concatenate((in_losses, out_losses))
                attack_test['member_status'] = np.array([1]*len(in_losses) + [0]*len(out_losses))
                attack_test['model_trajectory'] = np.concatenate((loss_traj_in.transpose(),loss_traj_out.transpose()))
                print("Building attack model")

                attack_train_set = torch.utils.data.TensorDataset(
                    torch.from_numpy(np.array(attack_data['model_loss_ori'], dtype='f')),
                    torch.from_numpy(np.array(attack_data['model_trajectory'], dtype='f')),
                    torch.from_numpy(np.array(attack_data['member_status'])).type(torch.long),)

                attack_test_set = torch.utils.data.TensorDataset(
                    torch.from_numpy(np.array(attack_test['model_loss_ori'], dtype='f')),
                    torch.from_numpy(np.array(attack_test['model_trajectory'], dtype='f')),
                    torch.from_numpy(np.array(attack_test['member_status'])).type(torch.long),)

                attack_train_loader = torch.utils.data.DataLoader(attack_train_set, batch_size=128, shuffle=True)
                attack_test_loader = torch.utils.data.DataLoader(attack_test_set, batch_size=128, shuffle=True)    

                attack_model = MLP_BLACKBOX(dim_in = attack_data['model_trajectory'].shape[1] + 1)
                attack_optimizer = torch.optim.SGD(attack_model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0001) 
                attack_model = attack_model.to(models.get_device(conf))
                loss_fn = torch.nn.CrossEntropyLoss()
                max_auc = 0
                max_acc = 0

                attack_fit(attack_model, attack_train_loader, attack_optimizer, loss_fn, models.get_device(conf), attack_test_loader=attack_test_loader, epochs=30)
                print("TrajectoryMIA")
                val_loss, val_prec1, traj_val_auc, _, _, tpr, fpr = test_mia_attack_model(attack_model, attack_test_loader, loss_fn, 100, 100, models.get_device(conf))
                print(val_loss, val_prec1, traj_val_auc)

                closest_index = [(np.abs(fpr - d)).argmin() for d in desired_fpr]
                traj_tpr_at_desired_fpr = tpr[closest_index]
                print(f"TPR@FPR{desired_fpr}:{traj_tpr_at_desired_fpr}")

            print("YEOM")
            pout, lout = models.predict_ds(client_model, test_ds, apply_softmax=False, conf=conf)
            pin, lin = models.predict_ds(client_model, cl_ds, apply_softmax=False, conf=conf)

            lsout = loss_function(torch.from_numpy(pout), torch.from_numpy(lout), reduction='none').detach().numpy()
            lsin = loss_function(torch.from_numpy(pin), torch.from_numpy(lin), reduction='none').detach().numpy()

            yeom_probs = np.concatenate((lsin, lsout))

            min_ = np.min(yeom_probs)
            max_ = np.max(yeom_probs) 
            yeom_probs = (yeom_probs-min_)/(max_-min_)
            yeom_probs = 1-yeom_probs
            yeom_labels = [1]*len(pin) + [0]*len(pout)
            Yeom_fpr, Yeom_tpr, thresholds = metrics.roc_curve(yeom_labels, yeom_probs)
            Yeom_roc_auc = metrics.auc(Yeom_fpr, Yeom_tpr)
            print(Yeom_roc_auc)
            closest_index = [(np.abs(Yeom_fpr - d)).argmin() for d in desired_fpr]
            yeom_tpr_at_desired_fpr = Yeom_tpr[closest_index]
            print(f"TPR@FPR{desired_fpr}:{yeom_tpr_at_desired_fpr}")
            
            print("LiRA")
            pout, lout = models.predict_ds(client_model, test_ds, apply_softmax=use_softmax, conf=conf)
            pin, lin = models.predict_ds(client_model, cl_ds, apply_softmax=use_softmax, conf=conf)
            pout = np.max(pout, -1)
            pin = np.max(pin, -1)
            if use_logit:
                pout = np.clip(pout, 1e-4, 1-1e-4)
                pin = np.clip(pin, 1e-4, 1-1e-4)
                pout = logit(pout)
                pin = logit(pin)
            import scipy
            pout_pred = scipy.stats.norm.cdf(pout, loc=pout_mean, scale=pout_std)
            pin_pred = scipy.stats.norm.cdf(pin, loc=pin_mean, scale=pin_std)
            lira_mlabel = [1]*len(pin_pred) + [0]*len(pout_pred)
            lira_probs_all_variance = np.concatenate((pin_pred, pout_pred))
            LiRA_fpr, LiRA_tpr, thresholds = metrics.roc_curve(lira_mlabel, lira_probs_all_variance)
            LiRA_roc_auc = metrics.auc(LiRA_fpr, LiRA_tpr)
            print(LiRA_roc_auc)
            closest_index = [(np.abs(LiRA_fpr - d)).argmin() for d in desired_fpr]
            lira_tpr_at_desired_fpr = LiRA_tpr[closest_index]
            print(f"TPR@FPR{desired_fpr}:{lira_tpr_at_desired_fpr}")        

            results = {"auc":{}, "tpr@fpr":{}}
            for i_fpr, fpr in enumerate(desired_fpr):
                results["tpr@fpr"][fpr] = {}
                if TRAJECTORY_ATTACK:
                    results["tpr@fpr"][fpr]["TrajectoryMIA"] = traj_tpr_at_desired_fpr[i_fpr]
                if LIRA_ATTACK:
                    results["tpr@fpr"][fpr]["LiRA"] = lira_tpr_at_desired_fpr[i_fpr]
                if YEOM_ATTACK:
                    results["tpr@fpr"][fpr]["Yeom"] = yeom_tpr_at_desired_fpr[i_fpr]
            results["auc"] = {}
            if TRAJECTORY_ATTACK:
                results["auc"]["TrajectoryMIA"] = traj_val_auc
            if LIRA_ATTACK:
                results["auc"]["LiRA"] = LiRA_roc_auc
            if YEOM_ATTACK:
                results["auc"]["Yeom"] = Yeom_roc_auc
            print(results)
        
            with open(
                os.path.join(args.root, model_id, f'post_attack_{client_id}.json'),
                "w",
            ) as f:
                f.write(json.dumps(results))
            if model_id not in all_results.keys():
                all_results[model_id] = {}
            all_results[model_id][client_id] = results
            print(all_results)

    print(all_results)
    for model_id in all_results.keys():
        with open(
            os.path.join(args.root, model_id, f'post_attack.json'),
            "w",
        ) as f:
            f.write(json.dumps(all_results[model_id]))

    if not os.path.exists("./dump/attack_log/"):
        os.makedirs("./dump/attack_log/")
    from datetime import datetime
    f_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(
        os.path.join("./dump/attack_log",f_name+".json"),
        "w",
    ) as f:
        f.write(json.dumps(all_results))
