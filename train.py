from qiskit.visualization import circuit_drawer
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_algorithms.optimizers import COBYLA

import time
import numpy as np
import os

from libs.circuit import Circuit


def train(epoch,
          train_images, train_labels, parameterized_train_labels,
          latent_bit_num, trash_bit_num,
          qstate_label_dict,
          savefolder,
          seed,
          ansatz_dict):

    np.random.seed(seed)
    
    parameter_savefolder = os.path.join(savefolder, "parameter")
    if not os.path.exists(parameter_savefolder):
        os.makedirs(parameter_savefolder)

    def cost_func_digits(ansatz_param):
        
        # パラメータを保存
        np.savetxt(os.path.join(parameter_savefolder, "{}.txt".format(len(os.listdir(parameter_savefolder)))), ansatz_param)

        probabilities = qnn.forward(np.hstack((parameterized_train_labels, train_images)), ansatz_param)
        cost = np.sum(probabilities[:, 1]) / train_images.shape[0]
        
        # 学習中の正解率と損失を保存
        with open(os.path.join(savefolder, "train_loss.txt"), "a+") as f:
            f.write(str(cost) + ',')
        
        return cost
   
    
    ### 回路を作成
    train_qc, input_params, weight_params = Circuit(latent_bit_num, trash_bit_num, ansatz_dict).train_circuit(len(set(train_labels)))

    circuit_drawer(train_qc, output='mpl', style="iqp", filename=os.path.join(savefolder, "qae_image_circuit_train.png"))
    circuit_drawer(train_qc, output='latex_source', filename=os.path.join(savefolder, "qae_image_circuit_train.tex"))

    with open(os.path.join(savefolder, "train_log.txt"), 'w') as f:
        f.write(f"Parameter num : {len(weight_params)}\n")
    
    def identity_interpret(x):
        return x

    # input_paramにデータエンコーディングを登録
    qnn = SamplerQNN(
        circuit=train_qc,
        input_params=input_params,
        weight_params=weight_params,
        interpret=identity_interpret,
        output_shape=2,
    )
    
    parameter_file_num = len(os.listdir(parameter_savefolder))
    
    epoch = epoch - parameter_file_num

    opt = COBYLA(maxiter=epoch)

    ## パラメータ初期値
    # 最初から学習する場合 → ランダムに設定
    if parameter_file_num == 0:
        initial_point = np.random.rand(len(weight_params))
    # 引き継いで学習する場合 → 最新のパラメータから読み込む
    else:
        initial_point = np.loadtxt(f"{parameter_savefolder}/{parameter_file_num - 1}.txt")

    # 学習（最適化関数の更新）
    start = time.time()
    opt_result = opt.minimize(fun=cost_func_digits, x0=initial_point)
    elapsed = time.time() - start
    print(f"Fit in {elapsed:0.2f} seconds")
    
    with open(os.path.join(savefolder, "train_log.txt"), 'a+') as f:
        f.write(f"\nFit in {elapsed:0.2f} seconds\n")
    
    