from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import transpile
from qiskit_aer import AerSimulator, QasmSimulator

import numpy as np
from tqdm import tqdm
from scipy.special import rel_entr


def eval_ansatz(ansatz, data_size, parameter_num, seed):

    n_qubits = ansatz.num_qubits

    ## Haarの算出

    # Possible Bin
    bins_list = []
    for i in range(76):
        bins_list.append((i)/75)

    # Center of the Bean
    bins_x = []    
    for i in range(75):
        bins_x.append(bins_list[1]+bins_list[i])

    def P_harr(l,u,N):
        return (1-l)**(N-1)-(1-u)**(N-1)
    
    # Harr historgram
    P_harr_hist=[]
    for i in range(75):
        P_harr_hist.append(P_harr(bins_list[i],bins_list[i+1],16))


    nshot = 10000
    nparam = 2000
    fidelity=[]

    for _ in tqdm(range(nparam)):

        input = np.random.rand(data_size)
        params = np.random.uniform(0, 2*np.pi, parameter_num) # パラメータをnparam分ランダムに生成

        target_ansatz = ansatz.assign_parameters(np.concatenate((input, params)))

        qr = QuantumRegister(n_qubits)
        cr = ClassicalRegister(n_qubits)
        qc = QuantumCircuit(qr, cr)
        qc = qc.compose(target_ansatz, range(n_qubits))
        qc.measure(qr[:],cr[:])

        simulator = AerSimulator(method='statevector')
        qct = transpile(qc, simulator) # シミュレータ用に回路を変換
        qct.save_statevector()
        job = simulator.run(qct, shots=nshot, seed_simulator=seed)

        result = job.result()
        
        count =result.get_counts()
        zeros = '0'*n_qubits
        if zeros in count and '1' in count:
            ratio=count[zeros]/nshot
        elif zeros in count and '1' not in count:
            ratio=count[zeros]/nshot
        else:
            ratio=0
        fidelity.append(ratio)


    weights = np.ones_like(fidelity)/float(len(fidelity))

    # example of calculating the kl divergence (relative entropy) with scipy
    P_hist = np.histogram(fidelity, bins=bins_list, weights=weights, range=[0, 1])[0]
    kl_pq = rel_entr(P_hist, P_harr_hist)
    express = sum(kl_pq)
    
    return express