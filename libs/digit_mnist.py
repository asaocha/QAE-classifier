import numpy as np 
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2

from .mnist_reader import load_mnist

def normalize(images):
    for i in range(len(images)):
        sum_sq = np.sum(images[i] ** 2)
        images[i] = images[i] / np.sqrt(sum_sq)
    return images

def parametrize_labels(labels, trash_bit_num, qstate_label_dict):
    
    # Rxゲートの設定が必要なビット数を取得
    for i in range(trash_bit_num):
        if 2**i <= len(set(labels)) <= 2**(i+1):
            rx_bit_num = i + 1
            break
    
    parameterized_labels_array = np.zeros((len(labels), rx_bit_num))  # (データ数, 判別用ビット数)
    for data_idx, label in enumerate(labels):
        qstate = [key for key, val in qstate_label_dict.items() if val == label][0]
        for qbit_idx, qstate_value in enumerate(reversed(qstate)):
            qstate_value = int(qstate_value)
            if qstate_value == 1:
                parameterized_labels_array[data_idx, qbit_idx] = np.pi
    return parameterized_labels_array


def get_mnist_dataset(label_list, train_data_num, test_data_num, 
                      qstate_label_dict,
                      trash_bit_num,
                      height, width,
                      seed, 
                      dataset="digit_mnist"):

    np.random.seed(seed)

    if dataset == "digit_mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset == "fashion_mnist":
        x_train, y_train = load_mnist('./data/fashion', kind='train')
        x_test, y_test = load_mnist('./data/fashion', kind='t10k')
    elif dataset == "kuzushiji_mnist":
        x_train, y_train = load_mnist('./data/kuzushiji', kind='train')
        x_test, y_test = load_mnist('./data/kuzushiji', kind='t10k')
    
        
    train_index, test_index = np.array([]), np.array([])
    # 特定のラベルのデータのみ抽出
    for label in label_list:
        train_index = np.append(train_index, np.where(y_train==label)[0]).astype(int)
        test_index = np.append(test_index, np.where(y_test==label)[0]).astype(int)

    # 指定枚数選択
    train_index = np.random.choice(train_index, train_data_num, replace=False)
    test_index = np.random.choice(test_index, test_data_num, replace=False)

    x_train = x_train[train_index]
    y_train = y_train[train_index]
    x_test = x_test[test_index]
    y_test = y_test[test_index]

    # リサイズ
    if dataset == "digit_mnist":
        x_train = np.array([cv2.resize(img, (width, height)) for img in x_train])/256
        x_test = np.array([cv2.resize(img, (width, height)) for img in x_test])/256
    elif dataset == "fashion_mnist" or dataset == "kuzushiji_mnist":
        x_train = np.array([cv2.resize(img, (width, height)) for img in np.reshape(x_train, (-1, 28, 28))])/256
        x_test = np.array([cv2.resize(img, (width, height)) for img in np.reshape(x_test, (-1, 28, 28))])/256

    # 画像をflatten
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)
    
    # ラベルからパラメータ行列を作成
    parametrized_y_train = parametrize_labels(y_train, trash_bit_num, qstate_label_dict)
    parametrized_y_test = parametrize_labels(y_test, trash_bit_num, qstate_label_dict)
        
    return x_train, y_train, parametrized_y_train, x_test, y_test, parametrized_y_test