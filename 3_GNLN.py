import os
import random
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import warnings

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.data_tensor = x
        self.target_tensor = y

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]


def softsign(x):
    f = x/(1+np.abs(x))

    return f

def uni_softsign(x):
    f = 0.5*softsign(x)+0.5
    return f


def uni_softsign_inverse(x):
    f = np.where(x >= 0.5,
                 -1. / (2. * x - 2) - 1 + 0.000000000000001,
                 1 - 1. / (2. * x + 0.000000001))
    return f

def softsign_activation(x):
    return x / (1 + np.absolute(x))

def x_2_softsign_activation_prime(x):
    X_less_than_1 = np.abs(x) < 1  
    X_greater_eq_than_1 = np.logical_not(X_less_than_1) 

    Y = np.zeros(x.shape)  

    Y[X_greater_eq_than_1] = 1

    Y[X_less_than_1] = 2 / (1 + np.absolute(x[X_less_than_1]))**2
    return Y


def softsign_activation_prime(x):
    return 2 * (1 - np.square(x)) / (1 + np.absolute(x))**2

def arcsine(x):
    return 2 / np.pi * np.arcsin(x)

def arcsine_prime(x):
    return 2 / np.pi / np.sqrt(1 - x**2 + 0.000000001)

def arcsine_softsign_prime(x):
    X_less_than_1 = np.abs(x) < 1  
    X_greater_eq_than_1 = np.logical_not(X_less_than_1)  

    Y = np.zeros(x.shape)  

    Y[X_less_than_1] = 2 / np.pi / np.sqrt(1 - x[X_less_than_1] ** 2)

    Y[X_greater_eq_than_1] = 1
    return Y

def MCC_calculate(output_temp, threshold, target):
    pred = torch.tensor(np.where(output_temp.cpu().numpy() > threshold, 1, 0).astype(int))
    # 计算TP, FP, FN, TN
    tp = torch.sum((target == 1) & (pred == 1)).item()
    fp = torch.sum((target == 0) & (pred == 1)).item()
    fn = torch.sum((target == 1) & (pred == 0)).item()
    tn = torch.sum((target == 0) & (pred == 0)).item()
    mcc = torch.tensor(tp * tn - fp * fn) / torch.sqrt(torch.tensor((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return mcc


def calculate_metrics(target, pred):
    """Calculate accuracy, recall, precision, and F1 score."""
    acc = accuracy_score(target, pred)
    recall = recall_score(target, pred)
    precision = precision_score(target, pred)
    f1 = f1_score(target, pred)
    return acc, recall, precision, f1


def plot_curve(x, y, xlabel, ylabel, title, text, filename):
    """Plot and save a curve (e.g., ROC, KS, PR)."""
    plt.figure()
    plt.plot(x, y)
    plt.text(0.5, 0.5, text)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(filename)


def accuracy_cal(output, target, output_auc, bestauc, best_sigma, key, get_tsv=0):
    correct = (output == target).sum().item()
    total = target.size(0)
    accuracy_best = 100 * correct / total

    true_positives = torch.sum((target == 1) & (output == 1))
    false_negatives = torch.sum((target == 1) & (output == 0))
    recall_best = 100 * true_positives / (true_positives + false_negatives)

    precision_best = 100 * torch.sum((target == 1) & (output == 1)) / (
            torch.sum((target == 1) & (output == 1)) + torch.sum((target == 0) & (output == 1)))

    if accuracy_best > 0.001:
        AUC = roc_auc_score(target.cpu().numpy(), output_auc.cpu().numpy())

        if accuracy_best + recall_best > 0:
            f1_best = 2 * (precision_best * recall_best) / (precision_best + recall_best)
        else:
            f1_best = 0

        if key == "Bagging_test":
            torch.save([target, output_auc], './figure_together/enhancement_target_output_auc.pth')
            if AUC > bestauc:
                bestauc = AUC
                best_sigma = sigma

        if key in ["Bagging_test", "Bagging_validate"]:
            os.makedirs(f'./{key}/', exist_ok=True)

            # ROC Curve
            fpr, tpr, thresholds = roc_curve(target.cpu().numpy(), output_auc.cpu().numpy())
            plot_curve(fpr, tpr, 'False Positive Rate', 'True Positive Rate', 'ROC Curve', f'AUC={AUC:.3f}',
                       f'./{key}/ROC.png')

            # Calculate metrics for each threshold
            metrics = np.array(
                [calculate_metrics(target.cpu().numpy(), (output_auc.cpu().numpy() > t).astype(int)) for t in
                 thresholds])

            acc_scores, recall_scores, precision_scores, F1_SCORES = metrics.T

            # Find best thresholds for each metric
            best_acc_idx = np.argmax(acc_scores)
            best_acc_threshold = thresholds[best_acc_idx]

            MCC = MCC_calculate(output_auc, best_acc_threshold, target.cpu())

            print(f"best_ACC {acc_scores[best_acc_idx] * 100:.2f}%",
                  f"corresponding_Recall {recall_scores[best_acc_idx] * 100:.2f}%",
                  f"corresponding_Precision {precision_scores[best_acc_idx] * 100:.2f}%",
                  f"corresponding_F1_scores {F1_SCORES[best_acc_idx] * 100:.2f}%",
                  f"corresponding_MCC {MCC:.3f}",
                  f"correspond_AUC {AUC:.3f}")

            # Find best thresholds for recall, precision, and F1 score
            best_recall_idx = np.argmax(recall_scores)
            best_precision_idx = np.argmax(precision_scores)
            best_F1_SCORES_idx = np.argmax(F1_SCORES)

            print(f"best_Recall {recall_scores[best_recall_idx]:.3f}",
                  f"best_precision {precision_scores[best_precision_idx]:.3f}",
                  f"best_F1_SCORES {F1_SCORES[best_F1_SCORES_idx]:.3f}")

            # KS Curve
            ks = np.max(tpr - fpr)
            plot_curve(thresholds, tpr - fpr, 'Threshold', 'Value', 'KS Curve', f'KS={ks:.3f}', f'./{key}/KS.png')

            # PR Curve
            precision, recall, _ = precision_recall_curve(target.cpu().numpy(), output_auc.cpu().numpy())
            ap = average_precision_score(target.cpu().numpy(), output_auc.cpu().numpy())
            plot_curve(recall, precision, 'Recall', 'Precision', 'PR Curve', f'AP={ap:.3f}', f'./{key}/PR.png')

            print(f"~~PR~~ {ap:.3f}")

            # Save to TSV if required
            if get_tsv == 1:
                tp = torch.sum((target == 1) & (output == 1)).item()
                fp = torch.sum((target == 0) & (output == 1)).item()
                fn = torch.sum((target == 1) & (output == 0)).item()
                tn = torch.sum((target == 0) & (output == 0)).item()

                # Calculate MCC
                mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

                index = np.array([f"{mcc:.4f}", f"{ap:.4f}", f"{ks:.4f}", f"{AUC:.4f}",
                                  f"{acc_scores[best_acc_idx]:.4f}", f"{recall_scores[best_acc_idx]:.4f}",
                                  f"{precision_scores[best_acc_idx]:.4f}", f"{F1_SCORES[best_acc_idx]:.4f}",
                                  f"{accuracy_best / 100:.4f}", f"{recall_best / 100:.4f}",
                                  f"{precision_best / 100:.4f}", f"{f1_best / 100:.4f}"])

                pd.DataFrame(index).to_csv("Ge_SAND_output.tsv", header=False, index=False, sep="\t")

        return accuracy_best, recall_best, precision_best, f1_best, AUC, bestauc, best_sigma
    else:
        return accuracy_best, recall_best, precision_best, 0, 0, bestauc, best_sigma
def train(w1, w2, epoch, train_dataloader, num_bag):
    random.seed(seed)

    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, data in enumerate(train_dataloader, 0):  
            inputs, target = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                target = target.cuda()

            hidden_temp = inputs.detach().cpu().numpy()
            label_temp = target.detach().cpu().numpy()

            input = hidden_temp @ w1

            input = softsign(input)

            ZNN_train = uni_softsign(input @ w2)

            E = ZNN_train - label_temp.reshape(-1, 1)

            ZNN_train = torch.tensor(ZNN_train, dtype=torch.double).cuda().reshape(-1)
            target = torch.tensor(target, dtype=torch.double).cuda()
            loss = criterion(ZNN_train, target)

            w2 = np.linalg.pinv(input) @ uni_softsign_inverse(
                 (E + label_temp.reshape(-1, 1)) - sigma[num_bag] * E/arcsine_softsign_prime(E))###softsign_activation_prime 0.5 tanh 0.6

            running_loss += loss.item()
            if batch_idx % 1 == 0 and epoch%100==0:
                print('Bag_num: %d, [%d, %5d] loss: %.3f' % (num_bag, epoch + 1, batch_idx + 1, running_loss / (800)))
    return w2


def test(w1, w2, AUC, dataloader_validate, num_bag, best_model, best, bestsigma):
    random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():  
        test_output_get = torch.tensor([]).to(device)
        test_target_output_get = torch.tensor([]).to(device)
        test_output_get_auc = torch.tensor([])
        for data in dataloader_validate:  
            images, labels = data  
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            hidden_temp = images.detach().cpu().numpy()
            input = hidden_temp @ w1
            input = softsign(input)
            ZNN_train = uni_softsign(input @ w2)
            ZNN_train = torch.tensor(ZNN_train, dtype=torch.double).cuda().reshape(-1)
            predicted = torch.round(ZNN_train)


            test_output_get_auc = torch.cat((test_output_get.to(torch.float), ZNN_train.to(torch.float)), dim=0)
            test_output_get = torch.cat((test_output_get.to(torch.float), predicted.to(torch.float)), dim=0)
            test_target_output_get = torch.cat((test_target_output_get.to(torch.float), labels.to(torch.float)), dim=0)

        accuracy_test, _, _, _, AUC_test, _, _ = accuracy_cal(test_output_get, test_target_output_get, test_output_get_auc, bestauc=best, best_sigma=bestsigma, key="test_for_training", get_tsv=0)


    if AUC < AUC_test:
        AUC = AUC_test
        model = OrderedDict([('w1', w1),('w2', w2)])
        torch.save(model, 'Bag_num_%d_FINETUNING_ZNN_model_classify_5e-5_SMOTE_model5.pth'%(num_bag))
        best_model = model


    return AUC, best_model, AUC_test


def train_and_test(n_bag, batch_size, best, bestsigma):
    random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    w1 = np.random.rand(n_bag, hidden_size, 64) 
    w2 = np.random.rand(n_bag, 64, 1)
    bag_sets = []
    validate_sets = []
    for i in range(int(n_bag)):
        validate_case_index = torch.arange(int(half_num_train_original + i * half_num_validate/n_bag), int(half_num_train_original + i * half_num_validate/n_bag + half_num_validate/n_bag))
        validate_control_index = torch.arange(int(half_num_train_original + batch_size/2 + i * half_num_validate/n_bag), int(half_num_train_original + batch_size/2 + i * half_num_validate/n_bag + half_num_validate/n_bag))

        validate_index = torch.cat([validate_case_index, validate_control_index], dim=0)

        this_validate_data = train_data[validate_index]
        this_validate_label = train_label[validate_index]

        train_index = torch.tensor([y for y in torch.arange(int(len(train_data))) if y not in validate_index])
        this_train_data = train_data[train_index]
        this_train_label = train_label[train_index]


        bag_sets.append((this_train_data, this_train_label))
        validate_sets.append((this_validate_data, this_validate_label))

    models = []
    num_bag = 0

    for bagging_number in range(len(bag_sets)):
        AUC = 0

        X_bag, Y_bag = bag_sets[bagging_number]
        dataset = MyDataset(X_bag, Y_bag)
        train_loader = DataLoader(dataset, batch_size=int(len(Y_bag)), shuffle=True)
        validate_X_bag, validate_Y_bag = validate_sets[bagging_number]
        validate_dataset = MyDataset(validate_X_bag, validate_Y_bag)
        validate_loader = DataLoader(validate_dataset, batch_size=int(len(validate_Y_bag)), shuffle=False)



        best_model = OrderedDict([('w1', w1[num_bag]), ('w2', w2[num_bag])])
        w2_temp = w2[num_bag]
        for epoch in range(1000):
            w2_temp = train(w1[num_bag], w2_temp, epoch, train_loader, num_bag)
            AUC, best_model, AUC_TEMP = test(w1[num_bag], w2_temp, AUC, validate_loader, num_bag, best_model, best, bestsigma)
            if AUC_TEMP == 0:
                break
        num_bag+=1
        best_model = [best_model['w1'], best_model['w2']]
        models.append(best_model)

    with torch.no_grad():
        test_output_get = torch.tensor([]).to(device)
        test_target_output_get = torch.tensor([]).to(device)
        test_output_get_auc = torch.tensor([]).to(device)
        for data in tensor_dataloader_test:
            features, labels = data
            features = features.to(device)
            labels = labels.to(device)
            y_pred = torch.tensor([]).to(device)
            for j in range(len(models)):
                w1_temp = models[j][0]
                w2_temp = models[j][1]

                hidden_temp = features.detach().cpu().numpy()
                input = hidden_temp @ w1_temp
                input = softsign(input)
                ZNN_train = uni_softsign(input @ w2_temp)
                ZNN_train = torch.tensor(ZNN_train, dtype=torch.double).cuda().reshape(-1)

                y_pred = torch.cat((y_pred.to(torch.float), ZNN_train.to(torch.float)), dim=0)

            y_pred = y_pred.reshape(n_bag, -1)

            percent_tage_set = torch.tensor([]).to(device)

            for i in range(y_pred.shape[1]):
                # print(y_pred)
                temp_label = y_pred[:, i].reshape(-1)
                count_label = torch.sum(temp_label)
                percent_tage = torch.tensor(count_label / n_bag).to(device).unsqueeze(0)

                percent_tage_set = torch.cat((percent_tage_set.to(torch.float), percent_tage.to(torch.float)))

            pred = torch.round(percent_tage_set)

            test_output_get = torch.cat((test_output_get.to(torch.float), pred.to(torch.float)), dim=0)
            test_target_output_get = torch.cat((test_target_output_get.to(torch.float), labels.to(torch.float)),
                                               dim=0)
            test_output_get_auc = torch.cat((test_output_get_auc.to(torch.float), percent_tage_set.to(torch.float)),
                                            dim=0)

        accuracy_test, _, _, _, _, best, best_sigma = accuracy_cal(test_output_get, test_target_output_get, test_output_get_auc,
                                                 bestauc=best, best_sigma=bestsigma, key="Bagging_test", get_tsv=1)



    torch.save(models, 'bagging_model_IDLN.pth')
    return best, best_sigma

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    torch.manual_seed(190807)  # Current CPU 190807

    torch.cuda.manual_seed(190807)  # Current GPU 190807
    seed = 190807
    hidden_size = 256
    random.seed(seed)
    train_data = torch.load("weights_token_mean_training.pth")
    num_train = int(len(train_data))
    half_num_train_original = int(num_train / 2)
    validate_data = torch.load("weights_token_mean_validating.pth")
    num_validate = int(len(validate_data))
    half_num_validate = int(num_validate / 2)

    train_data = torch.cat(
        [train_data[:half_num_train_original], validate_data[:half_num_validate], train_data[half_num_train_original:],
         validate_data[half_num_validate:]], dim=0)
    num_train = int(len(train_data))
    half_num_train = int(num_train / 2)
    train_label = torch.cat([torch.ones(half_num_train), torch.zeros(half_num_train)])

    test_data = torch.load("weights_token_mean_testing.pth")
    num_test = int(len(test_data))
    half_num_test = int(num_test / 2)
    test_label = torch.cat([torch.ones(half_num_test), torch.zeros(half_num_test)])

    my_dataset_train = MyDataset(train_data, train_label)

    my_dataset_test = MyDataset(test_data, test_label)

    tensor_dataloader_train = DataLoader(
        dataset=my_dataset_train,
        batch_size=num_train,
        shuffle=True,
        num_workers=0
    )
    print('tensor_dataloader_train:', tensor_dataloader_train)

    tensor_dataloader_test = DataLoader(
        dataset=my_dataset_test,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    criterion = nn.BCELoss(reduction='sum')
    random.seed(seed)

    best_test = 0#
    best_sigma = 0
    sigma = [0.3, 0.1]  #
    lr = 0
    n_bag = 2
    batch_size = int(num_train)  #700+700
    best_test, best_sigma = train_and_test(n_bag, batch_size, best_test, best_sigma)
    print("Now", sigma)
    print(best_test, best_sigma)





