import numpy as np

from src.models import predict_losses, predict, calculate_loss


def yeom_mi_attack(losses, threshold):
    """YEOM et all's membership inference attack using pred loss"""
    memberships = (losses < threshold).astype(int)
    return memberships


def get_af():
    return yeom_mi_attack


def attacker_observation(model, attacker_knowledge, loss_function, *args, **kwargs):
    """Attacker tests model with it's train data knowledge"""
    train_attacker = attacker_knowledge["in_train_data"]
    test_attacker = attacker_knowledge["not_train_data"]
    x_train_attacker, y_train_attacker = train_attacker
    x_test_attacker, y_test_attacker = test_attacker

    loss_train_attacker = predict_losses(
        model, x_train_attacker, y_train_attacker, loss_function, *args, **kwargs
    )
    loss_test_attacker = predict_losses(
        model, x_test_attacker, y_test_attacker, loss_function, *args, **kwargs
    )

    return loss_train_attacker, loss_test_attacker


def attack(model, attacker_knowledge, mia_data, loss_function, *args, **kwargs):
    """Attacker performs attack"""

    loss_train_attacker, loss_test_attacker = attacker_observation(
        model, attacker_knowledge, loss_function, *args, **kwargs
    )
    cls_threshold = yeom_classwise_threshold(
        model, attacker_knowledge, loss_function, *args, **kwargs
    )
    avg_threshold = yeom_standard_threshold(loss_train_attacker)
    best_threshold = yeom_best_threshold(loss_train_attacker, loss_test_attacker)

    x_mia_data, y_mia_data = mia_data
    loss_mia = predict_losses(
        model, x_mia_data, y_mia_data, loss_function, *args, **kwargs
    )

    attack_preds = {}
    attack_preds["adv_std"] = yeom_mi_attack(loss_mia, avg_threshold)
    attack_preds["adv_pow"] = yeom_mi_attack(loss_mia, best_threshold)
    attack_preds["adv_cls"] = classwise_attack(loss_mia, cls_threshold, y_mia_data)

    return attack_preds


def calculate_advantage(y_true, y_pred):
    """Two times the advantage over 50% random guess to get a 0-100 percentage-like result"""
    return 2 * (len(y_true[y_true == y_pred]) - len(y_true) / 2) / len(y_true) * 100


def yeom_standard_threshold(train_losses):
    return np.mean(train_losses)


def yeom_best_threshold(train_losses, not_train_losses):
    """Get threshold for Sablayrolles et al"""
    advantages = []

    mean_loss = np.mean(train_losses)
    std_dev = np.std(train_losses)

    x = np.concatenate((train_losses, not_train_losses))
    y_true = [1.0] * len(train_losses) + [0.0] * len(not_train_losses)
    y_true = np.array(y_true)

    coeffs = np.linspace(-5, 5, num=1001, endpoint=True)

    for coeff in coeffs:
        cur_threshold = mean_loss + std_dev * coeff
        cur_pred = yeom_mi_attack(x, cur_threshold)
        cur_yeom_mi_advantage = calculate_advantage(y_true, cur_pred)
        advantages.append(cur_yeom_mi_advantage)

    best_threshold = mean_loss + std_dev * coeffs[np.argmax(advantages)]

    return best_threshold


def yeom_classwise_threshold(model, attacker_knowledge, loss_function, *args, **kwargs):
    """Avg for each class"""
    train_attacker = attacker_knowledge["in_train_data"]
    test_attacker = attacker_knowledge["not_train_data"]
    x_train_attacker, y_train_attacker = train_attacker
    x_test_attacker, y_test_attacker = test_attacker
    thresholds = {}
    for cls in np.unique(y_train_attacker):
        x_train_attacker_cls = x_train_attacker[y_train_attacker == cls]
        y_train_attacker_cls = y_train_attacker[y_train_attacker == cls]
        y_pred_cls = predict(model, x_train_attacker_cls, verbose=0)
        loss_train_attacker_cls = calculate_loss(y_train_attacker_cls, y_pred_cls, loss_function, reduction='mean')
        thresholds[cls] = loss_train_attacker_cls
    return thresholds


def classwise_attack(loss_mia, thresholds, y_mia_data):
    pred = np.ones(len(y_mia_data), dtype=bool)
    for cls in thresholds.keys():
        cls_idx = np.where(y_mia_data == cls)
        l = loss_mia[cls_idx]
        mia_pred = yeom_mi_attack(l, thresholds[cls])
        pred[cls_idx] = mia_pred
    return pred
