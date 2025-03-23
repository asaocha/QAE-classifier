import numpy as np
from scipy import linalg
from sklearn.metrics import accuracy_score
from scipy.special import softmax
import cv2

G = 9  # ラベルを表すg-hotの要素

class SVD:
    def __init__(self, label_num, k, seed):
        self.label_num = label_num
        self.k = k
        self.seed = seed

        np.random.seed(seed)

    def train(self, train_images, train_labels, training_epochs):

        X = self.image_array(train_images, train_labels)

        U, S, V = linalg.svd(X)

        U_reduced = U[:, :self.k]
        S_reduced = np.matrix(linalg.diagsvd(S[:self.k], self.k, self.k))
        V_reduced = V[:self.k, :]

        S_root = linalg.sqrtm(S_reduced)

        reconst_reduced = np.linalg.multi_dot([U_reduced, S_root, S_root, V_reduced])

        self.calc_accuracy(reconst_reduced[-self.label_num:], train_labels)

        return np.dot(U_reduced, S_root), np.dot(S_root, V_reduced)


    def test(self, test_images, test_labels, W):

        V_test = test_images.copy().T

        w1, w2 = np.split(W, [-self.label_num])

        H_test = np.dot(np.linalg.pinv(w1), V_test)
        U = softmax(np.dot(w2, H_test), axis=0)

        return self.calc_accuracy(U, test_labels)


    def image_array(self, images, labels):
        images = images.copy().T
        label_array = np.zeros((self.label_num, images.shape[1])) # ラベル代入用の行列
        for i in range(images.shape[1]):
            label = labels[i]
            label_array[label, i] = G
        return np.vstack((images, label_array))

    def calc_accuracy(self, label_array, true_label_list):
        predict_label_list = []
        for i in range(label_array.shape[1]):
            predict_label_list.append(np.argmax(label_array[:,i]))
        return accuracy_score(true_label_list, predict_label_list)

