from qiskit import transpile
from qiskit_aer import QasmSimulator

import numpy as np
from tqdm import tqdm


def eval_ansatz(ansatz, data_size, parameter_savefile_path, seed, samples=1024):
    np.random.seed(seed)
    pqc_integral_value = pqc_integral(ansatz, samples, seed, data_size, parameter_savefile_path)
    expressibility = np.linalg.norm(haar_integral(ansatz.num_qubits, samples) - pqc_integral_value)
    return expressibility
    

def random_unitary(N):
    Z = np.random.randn(N, N) + 1.0j * np.random.randn(N, N)
    [Q, R] = np.linalg.qr(Z)
    D = np.diag(np.diagonal(R) / np.abs(np.diagonal(R)))
    return np.dot(Q, D)

def haar_integral(num_qubits, samples):
    N = 2**num_qubits
    randunit_density = np.zeros((N, N), dtype=complex)

    zero_state = np.zeros(N, dtype=complex)
    zero_state[0] = 1
    
    for _ in tqdm(range(samples)):
        A = np.matmul(zero_state, random_unitary(N)).reshape(-1,1)
        randunit_density += np.kron(A, A.conj().T) 

    randunit_density/=samples

    return randunit_density
    
def pqc_integral(ansatze, samples, seed, data_size, parameter_savefile_path):
    N = ansatze.num_qubits
    randunit_density = np.zeros((2**N, 2**N), dtype=complex)

    for _ in tqdm(range(samples)):
        input = np.random.rand(data_size)
        params = np.loadtxt(parameter_savefile_path)
        ansatz = ansatze.assign_parameters(np.concatenate((input, params)))

        simulator = QasmSimulator(method='statevector')
        qct = transpile(ansatz, simulator) # シミュレータ用に回路を変換
        job = simulator.run(qct, shots=1024, seed_simulator=seed)
        U = np.asarray(job.result().data(0)['v1']).reshape(-1,1)

        ### 表現力
        randunit_density += np.kron(U, U.conj().T)

    return randunit_density/samples