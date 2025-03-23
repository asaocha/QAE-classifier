# from qiskit import execute, Aer
from qiskit_aer import QasmSimulator
from qiskit.visualization import circuit_drawer
from qiskit import transpile

import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import os

from libs.circuit import Circuit
from libs.plot_figure import prediction_certainty_hist, prediction_heatmap


def classification(test_images, test_labels, label_list,
                   latent_bit_num, trash_bit_num,
                   qstate_label_dict,
                   savefolder, parameter_savefile_path,
                   seed,
                   ansatz_dict):
    
    test_qc = Circuit(latent_bit_num, trash_bit_num, ansatz_dict).classification_circuit()
    circuit_drawer(test_qc, output='mpl', style="iqp", filename=os.path.join(savefolder, "qae_image_circuit_test.png"))
    circuit_drawer(test_qc, output='latex_source', filename=os.path.join(savefolder, "qae_image_circuit_test.tex"))
    
    circuit_param = np.loadtxt(parameter_savefile_path)
    
    predict_labels = []
    true_certainty_list, false_certainty_list = [], []
    pred_array = np.zeros((len(test_images), len(label_list)))
    
    for i, image in enumerate(test_images):
        param_values = np.concatenate((image, circuit_param))
        output_qc = test_qc.assign_parameters(param_values)
        
        # job = execute(output_qc, Aer.get_backend('qasm_simulator'), shots=1024, seed_simulator=seed)
        # result = job.result().get_counts(output_qc)
        
        simulator = QasmSimulator(method='statevector')
        # simulator = QasmSimulator(method='statevector', device='GPU')
        qct = transpile(output_qc, simulator) # シミュレータ用に回路を変換
        job = simulator.run(qct, shots=1024, seed_simulator=seed)
        result = job.result().get_counts(output_qc)
        
        probabilities = {key: value / sum(result.values()) for key, value in result.items()}
        for key, prob in probabilities.items():
            if qstate_label_dict[key] in label_list:
                pred_array[i, qstate_label_dict[key]] = prob
            
        
        # 最も測定確率が高かった状態をラベルに変換
        prediction = qstate_label_dict[max(result, key=result.get)]
        predict_labels.append(prediction)
        
        # 予測の確信度
        certainty = max(result.values())/sum(result.values())
        if prediction == test_labels[i]:
            true_certainty_list.append(certainty)
        else:
            false_certainty_list.append(certainty)
    
    accuracy = accuracy_score(test_labels, predict_labels)
    
    log_loss_value = log_loss(np.array(test_labels), pred_array/pred_array.sum(axis=1, keepdims=True))
            
    print("test labels : ", test_labels)
    print("predict labels : ", predict_labels)
    print(accuracy)
    print('log loss : ', log_loss_value)
    
    with open(os.path.join(savefolder, "test_log.txt"), 'w') as f:
        f.write(f'accuracy : {accuracy*100:.2f}%\n')
        f.write(f"test labels : {test_labels}\n")
        f.write(f"predict labels : {predict_labels}\n")
        f.write(f"log loss : {log_loss_value}\n")
    
    prediction_certainty_hist(true_certainty_list, false_certainty_list, test_labels, "test", savefolder)
    
    prediction_heatmap(test_labels, predict_labels, label_list, savefolder)
