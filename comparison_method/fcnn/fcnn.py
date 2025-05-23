import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import utils
import tensorflow_model_optimization as tfmot
import numpy as np

def print_layer_sparsity(model):
    print("▼ スパース率（ゼロの重みの割合）:")
    total_params = 0
    total_zeros = 0
    
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:  # 重みを持つ層だけ処理
            w = weights[0]  # 通常 weights[0] が kernel（重み）
            zero_count = np.sum(w == 0)
            total_size = w.size
            sparsity = zero_count / total_size
            
            # バイアスがある場合はそれも計算に含める
            if len(weights) > 1:
                b = weights[1]  # バイアス
                zero_count_bias = np.sum(b == 0)
                zero_count += zero_count_bias
                total_size += b.size
            
            total_params += total_size
            total_zeros += zero_count
            
            print(f"{layer.name:<30} : {sparsity:.2%} ゼロ ({zero_count}/{total_size} パラメータ)")
    
    # モデル全体のスパース率と0の数を表示
    overall_sparsity = total_zeros / total_params if total_params > 0 else 0
    print("\n▼ モデル全体:")
    print(f"総パラメータ数: {total_params}")
    print(f"ゼロの重みの数: {total_zeros}")
    print(f"実際のパラメータ数: {total_params - total_zeros}")
    print(f"全体のスパース率: {overall_sparsity:.2%}")
    
    return total_zeros, total_params

class FCNN:
    def __init__(self):
        self.model = None

    def train(self, 
              train_images, train_labels, 
              training_epochs,
              label_num, k,
              sparsity=0.98, frequency=1):
        
        input_size = int(np.sqrt(train_images.shape[1]))

        train_images = train_images.reshape(-1, input_size, input_size).copy()
        train_labels_categorical = utils.to_categorical(train_labels)

        # pruning設定
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=sparsity,   # ← 70% のスパース率
                begin_step=0,          # トレーニング直後から開始
                frequency=frequency          # マスクの更新頻度（ステップごと）
            )
        }   

        self.model = tf.keras.Sequential([
            Flatten(input_shape=(input_size, input_size)),
            prune_low_magnitude(Dense(k, activation='sigmoid'), **pruning_params),
            prune_low_magnitude(Dense(label_num, activation='softmax'), **pruning_params)
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
        self.model.fit(train_images, train_labels_categorical, epochs=training_epochs, validation_split=0.33, callbacks=callbacks)
        self.model = tfmot.sparsity.keras.strip_pruning(self.model)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # スパース率の確認
        print_layer_sparsity(self.model)

        print(self.model.summary())


    
    def test(self, test_images, test_labels):

        input_size = int(np.sqrt(test_images.shape[1]))
        
        test_images = test_images.reshape(-1, input_size, input_size).copy()
        test_labels_categorical = utils.to_categorical(test_labels)

        _, test_accuracy = self.model.evaluate(test_images, test_labels_categorical)
        
        return test_accuracy
