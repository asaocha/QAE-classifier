import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


def prediction_certainty_hist(true_certainty_list, false_certainty_list, test_labels, type, savefolder):
    
    plt.figure(figsize=(14,8))
    plt.hist(true_certainty_list, label="True prediction", bins=50)
    plt.hist(false_certainty_list, label="False prediction", bins=50)
    plt.xlabel("Certainty", fontsize=24); plt.ylabel("Frequency", fontsize=24)
    # plt.xlim(1/len(list(set(test_labels))),1)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(savefolder, "{}_uncertainty.jpg".format(type)))
    plt.clf()
    plt.close()
    

def prediction_heatmap(true_labels, pred_labels, label_list, savefolder):
    
    # label_list.insert(0, -1)

    # 正解ラベルと予測ラベルから混同行列を作成する
    cm = np.zeros((len(label_list), len(label_list)))
    for true, pred in zip(true_labels, pred_labels):
        if pred == -1: # -1予測は除外
            continue
        cm[label_list.index(true), label_list.index(pred)] += 1

    # ヒートマップを描画する
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=label_list, yticklabels=label_list, square=True, ax=ax)

    # 軸のラベルを設定する
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    # 図を表示する
    plt.tight_layout()
    plt.savefig(os.path.join(savefolder, "prediction_heatmap.jpg"))
    plt.clf()
    plt.close()


def loss(train_loss, savefolder):
    plt.clf()
    plt.figure(figsize=(12,8))
    plt.plot(train_loss)
    plt.xlabel("Train epochs", fontsize=18)
    plt.ylabel("Objective function value", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(savefolder, "train_loss.jpg"))
    plt.clf()
    plt.close() 
    
def loss_and_accuracy(train_loss, test_accuracy, train_accuracy, savefolder):
    fig, ax1 = plt.subplots(figsize=(14,8))
    ax1.plot(range(len(train_loss)), train_loss, "r-", label="Objective function value")
    ax1.set_ylabel("Objective function value", fontsize=24)
    plt.tick_params(labelsize=18)
    
    ax2 = ax1.twinx() 
    ax2.plot(range(len(test_accuracy)), test_accuracy, "b-", label="Test accuracy")
    ax2.plot(range(len(train_accuracy)), train_accuracy, "g-", label="Train accuracy")
    ax2.set_ylabel("Accuracy", fontsize=24)
    plt.tick_params(labelsize=18)
    
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='center right', fontsize=18)
    ax1.set_xlabel("Train epochs", fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(savefolder, "loss_and_accuracy.jpg"))
    plt.clf()
    plt.close() 
    
    
if __name__ == '__main__':
    savefolder = "/workspace/qiskit/qae-classifier/output/train_result_20241027_132502"
    with open(os.path.join(savefolder, "train_loss.txt"), "r") as f:
        train_loss = np.array(f.read().strip(",").split(","), dtype=float)
    loss(train_loss, savefolder)