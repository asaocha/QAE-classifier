import numpy as np
from pyqubo import solve_qubo

class Annealing:
    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def simulated_annealing(self, V, W):

        k = W.shape[1]
        m = V.shape[1]

        H = np.zeros((k, m))

        for x in range(m):

            v =  V[:,x]

            # QUBO(k,k)に変換
            diagonal_matrix = np.zeros(k)
            for i in range(k):
                diagonal_matrix[i] = np.dot(W[:,i], W[:,i] - 2*v)

            upper_tri_matrix = np.zeros((k, k))
            for i in range(k):
                for j in range(i+1, k):
                    upper_tri_matrix[i, j] = 2*np.dot(W[:,i], W[:,j])

            qubo_matrix = np.diag(diagonal_matrix + self.alpha) + upper_tri_matrix

            qubo_dict = {}
            for i in range(qubo_matrix.shape[0]):
                for j in range(qubo_matrix.shape[0]):
                    qubo_dict[(f'q[{str(i)}]', f'q[{str(j)}]')] = qubo_matrix[i][j]

            solution_dict = solve_qubo(qubo_dict)
            # SAのパラメータ 要確認

            # q[i]で返ってくる解を1/0で行列に出力する
            for i in range(k):
                H[i,x] = int(solution_dict[f'q[{str(i)}]'])

        return H