import numpy as np

RMSPROP_ITER = 100  # 試行回数

# RMSPorpの実装
class RMSProp:

    # インスタンス変数を定義
    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr # 学習率
        self.decay_rate = decay_rate # 減衰率
        self.h = None # 過去の勾配の2乗和
    
    # パラメータの更新メソッドを定義
    def update(self, params, grads):
        # hの初期化
        if self.h is None: # 初回のみ
            self.h = np.zeros_like(params)
        
        # パラメータの値を更新
        for i in range(params.shape[0]):
            self.h[i] *= self.decay_rate # 式(1)の前の項
            self.h[i] += (1 - self.decay_rate) * grads[i] * grads[i] # 式(1)の後の項
            params[i] -= self.lr * grads[i] / (np.sqrt(self.h[i]) + 1e-7) # 式(2)
        
        return params


# 勾配 ▽f(x)
def grad(x, v, H, alpha):
    return (-1)*np.dot(H, v - np.dot(H.T, x)) + alpha*np.sum(x)

# P[x]
def p(optimizer, x, v, H, u, alpha):
    y = optimizer.update(x, grad(x, v, H, alpha))
    y = np.where(y > 0, y, 0).copy()
    y = np.where(y < u, y, u).copy()
    return y


def calc(V, H, seed, alpha=0.0001, lr=0.01, decay_rate=0.99):

    # np.random.seed(seed)

    u = 1.0

    n = V.shape[0]
    k = H.shape[0]

    # パラメータの初期値を指定
    W_init = np.random.uniform(0, u, (n, k))
    W = np.zeros_like(W_init)

    for i in range(n):

        x = W_init[i]
        v = V[i]

        optimizer = RMSProp(lr=lr, decay_rate=decay_rate)

        # 関数の最小値を探索
        for _ in range(RMSPROP_ITER):

            x2 = p(optimizer, x, v, H, u, alpha) # x^k+1
            x = x2.copy() # xを更新

            loss = np.linalg.norm(v - np.dot(x, H), 2)

            if _ == 0:
                min_loss = loss
                x_min = x.copy()

            # 最小値に更新
            if min_loss > loss:
                min_loss = loss
                x_min = x.copy()

            if (x_min < 0).any():
                print(np.where(x_min < 0))

        W[i] = x_min

    return W