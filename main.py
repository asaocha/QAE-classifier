from train import train
from test import classification

from libs.digit_mnist import get_mnist_dataset
from libs.label import possible_qstate
from libs.plot_figure import loss

from comparison_method.nbmf.nbmf import NBMF
from comparison_method.fcnn.fcnn import FCNN
from comparison_method.svd.svd import SVD

import os, shutil
import sys
from datetime import datetime
import numpy as np
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--load_parameter', action='store_true')
parser.add_argument('--savefolder', default='None')
parser.add_argument('--skip_training', action='store_true')
parser.add_argument('--load_epoch', default=-1, type=int)

args = parser.parse_args()


# あらかじめ学習したものを読み込む場合：フォルダを指定
if args.load_parameter:
    savefolder = args.savefolder
    train_parameter_json_filepath = os.path.join(savefolder, "train_parameter.json")

# 新しく学習する場合：保存先を作る
else:
    savefolder = "output/train_result_{}".format(datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(savefolder, exist_ok=True)
    train_parameter_json_filepath = "train_parameter.json"
    shutil.copy2(train_parameter_json_filepath, savefolder)

### 学習時のパラメータ 読込
with open(train_parameter_json_filepath, 'r') as f:
    parameter_d = json.load(f)["parameter"]
    
seed = parameter_d["seed"]
label_list = parameter_d["label_list"]
trash_bit_num = parameter_d["trash_bit_num"]
train_data_num = parameter_d["train_data_num"]
test_data_num = parameter_d["test_data_num"]
height = parameter_d["height"]
width = parameter_d["width"]
epoch = parameter_d["epoch"]
ansatz_dict = parameter_d["ansatz_dict"]

if 'label_type' in parameter_d:
    label_type = parameter_d["label_type"]
else:
    label_type = "compressed"


# 潜在状態のビット数：RawFeatureVectorの場合、画素数とラベル数に応じる
latent_bit_num = int(np.log2(height*width)) - trash_bit_num

# ビット数から識別可能なラベルを取得
qstate_label_dict = possible_qstate(trash_bit_num, label_list, label_type)
print(qstate_label_dict)


### MNISTデータセット作成
train_images, train_labels, parameterized_train_labels, test_images, test_labels, _ = get_mnist_dataset(label_list, train_data_num, test_data_num, qstate_label_dict, trash_bit_num, height, width, label_type, seed)

## QAE分類
if not 'method' in parameter_d:
    # 学習
    if not args.skip_training:
        train(epoch, 
            train_images, train_labels, parameterized_train_labels, 
            latent_bit_num, trash_bit_num,
            qstate_label_dict,
            savefolder,
            seed,
            ansatz_dict)
        
    # テストに使うパラメータ
    parameter_savefolder = f'{savefolder}/parameter'
    if args.load_epoch == -1:  # 最新パラメータ
        parameter_savefile_path = f'{parameter_savefolder}/{len(os.listdir(parameter_savefolder)) - 1}.txt'
    else:  # 指定のエポックのパラメータ
        parameter_savefile_path = f'{parameter_savefolder}/{args.load_epoch}.txt'

    # テスト    
    classification(test_images, test_labels, label_list,
                latent_bit_num, trash_bit_num, 
                qstate_label_dict,
                savefolder, parameter_savefile_path,
                seed,
                ansatz_dict)
        
    # 損失関数のグラフ
    with open(os.path.join(savefolder, "train_loss.txt"), "r") as f:
        train_loss = np.array(f.read().strip(",").split(","), dtype=float)
        loss(train_loss, savefolder)

## 比較手法
else:
    method = parameter_d['method']

    k = int(2**latent_bit_num)

    if method == 'nbmf':
        nbmf = NBMF(len(label_list), k, seed)
        W, H = nbmf.train(train_images, train_labels, epoch)
        test_accuracy = nbmf.test(test_images, test_labels, W)
    elif method == 'fcnn':
        fcnn = FCNN()
        fcnn.train(train_images, train_labels, epoch, len(label_list), k)
        test_accuracy = fcnn.test(test_images, test_labels)
    elif method == 'svd':
        svd = SVD(len(label_list), k, seed)
        W, H = svd.train(train_images, train_labels, epoch)
        test_accuracy = svd.test(test_images, test_labels, W)
    
    print('test accuracy : ', test_accuracy)