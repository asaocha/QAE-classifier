
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from sklearn.metrics import accuracy_score

from .libs import proj_rmsprop
from .libs import annealing

G = 9  # ラベルを表すg-hotの要素

class NBMF:
    def __init__(self, label_num, k, seed):
        self.label_num = label_num
        self.k = k
        self.seed = seed

        np.random.seed(seed)

        self.annealer = annealing.Annealing()

    def train(self, train_images, train_labels, training_epochs):

        V = self.image_array(train_images, train_labels)

        H = np.random.randint(0, 2, (self.k, V.shape[1]))

        for _ in tqdm(range(training_epochs)):

            ## W 更新 ##
            W = np.zeros((V.shape[0], self.k))  # 初期化
            W = proj_rmsprop.calc(V, H, self.seed)  # 更新

            ## H 更新 ##
            H = self.annealer.simulated_annealing(V, W)

            # バリデーション
            validation_accuracy = self.test(train_images, train_labels, W)

        print('validation accuracy : ', validation_accuracy)

        total_params = W.shape[0]*W.shape[1] + H.shape[0] # パラメータ数
        total_zeros = int(W.shape[0]*np.mean(np.sum(H == 0, axis=0))) # Hが0 = 使われなかったパラメータ数 (Hの列平均から算出)
        sparsity = total_zeros / total_params
        print('スパース性 : ', sparsity)
        print(f"総パラメータ数: {total_params}")        
        print(f"実際のパラメータ数: {total_params - total_zeros}")

        return W, H


    def test(self, test_images, test_labels, W):

        V_test = test_images.copy().T

        w1, w2 = np.split(W, [-self.label_num])

        H_test = self.annealer.simulated_annealing(V_test, w1)
        U = softmax(np.dot(w2, H_test), axis=0)

        pred_labels = []
        for i in range(U.shape[1]):
            pred_labels.append(np.argmax(U[:,i]))

        return accuracy_score(test_labels, pred_labels)

    
    def image_array(self, images, labels):
        images = images.copy().T
        label_array = np.zeros((self.label_num, images.shape[1])) # ラベル代入用の行列
        for i in range(images.shape[1]):
            label = labels[i]
            label_array[label, i] = G
        return np.vstack((images, label_array))