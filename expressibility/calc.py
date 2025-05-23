from libs.ansatz import Ansatz
from expressibility.func import eval_ansatz

import argparse
import os
import json
import numpy as np

from qiskit import QuantumCircuit
from qiskit_machine_learning.circuit.library import RawFeatureVector

SEED = 123    

parser = argparse.ArgumentParser()
parser.add_argument('--savefolder', default='None')
args = parser.parse_args()

train_parameter_json_filepath = os.path.join(args.savefolder, "train_parameter.json")

### 学習時のパラメータ 読込
with open(train_parameter_json_filepath, 'r') as f:
    parameter_d = json.load(f)["parameter"]
    
seed = parameter_d["seed"]
trash_bit_num = parameter_d["trash_bit_num"]
height = parameter_d["height"]
width = parameter_d["width"]
ansatz_dict = parameter_d["ansatz_dict"]

print(ansatz_dict)

# 潜在状態のビット数：RawFeatureVectorの場合、画素数とラベル数に応じる
latent_bit_num = int(np.log2(height*width)) - trash_bit_num
num_qubits = latent_bit_num + trash_bit_num

# アンザッツ作成
input_circuit = QuantumCircuit(num_qubits)
feature_mapping = RawFeatureVector(2**num_qubits)
data_size = 2**num_qubits
target_ansatz = Ansatz(ansatz_dict, latent_bit_num, trash_bit_num).ansatz
input_circuit = input_circuit.compose(feature_mapping, range(num_qubits))
input_circuit = input_circuit.compose(target_ansatz, range(num_qubits))
input_circuit.save_statevector(label='v1')

parameter_num = target_ansatz.num_parameters

expressibility = eval_ansatz(input_circuit, data_size, parameter_num, SEED)
print(f'seed:{seed}, expressibility:{expressibility}')

