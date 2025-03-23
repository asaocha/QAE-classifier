from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.circuit.library import RawFeatureVector

from .ansatz import Ansatz


### クラスにする
# latent_bit_num, trash_bit_num, ansatz_name, reps, entanglementをクラス変数に
class Circuit:
    def __init__(self, latent_bit_num, trash_bit_num, ansatz_dict):
        self.latent_bit_num = latent_bit_num
        self.trash_bit_num = trash_bit_num
        self.ansatz_dict = ansatz_dict
    

    def auto_encoder_circuit(self):
        
        # 潜在空間のビット + トラッシュ空間のビット + 参照空間のビット + スワップテスト用のancillaビット
        qr = QuantumRegister(self.latent_bit_num + 2 * self.trash_bit_num + 1, "q")
        
        # スワップテストの結果測定用の古典ビット
        cr = ClassicalRegister(1, "c")
        
        # レジスタから量子回路作成
        circuit = QuantumCircuit(qr, cr)
        
        # ansatzを組み込む
        circuit.compose(Ansatz(self.ansatz_dict, self.latent_bit_num, self.trash_bit_num).ansatz, 
                        range(0, self.latent_bit_num + self.trash_bit_num), inplace=True)
        # inplace=True：破壊的に変更される。ansatz自体の内容が変更される
        # inplace=False：ansatz自体の情報が保持されたまま、コピーが組み込まれる
        
        circuit.barrier()
        
        # swap test
        ancilla_qubit = self.latent_bit_num + 2 * self.trash_bit_num   # スワップテスト用ビットの番号
        circuit.h(ancilla_qubit)
        for i in range(self.trash_bit_num):
            # (スワップテスト用, トラッシュ空間のビット, 参照空間のビット)
            circuit.cswap(ancilla_qubit, self.latent_bit_num + i, self.latent_bit_num + self.trash_bit_num + i)
        circuit.h(ancilla_qubit)
        
        # 測定
        # 古典で行う理由：1ビットの情報しか持たず、古典レジスタで測定しても情報量はほとんど変わらないため。量子ビットの測定回数を節約する
        circuit.measure(ancilla_qubit, cr[0])
        
        return circuit


    def train_circuit(self, label_num):
        input_bit = self.latent_bit_num + self.trash_bit_num
        
        # 入力ビット分の状態ベクトルを作成
        # 特徴量ベクトルを入力として受け取り、それを量子状態にエンコード
        feature_mapping = RawFeatureVector(2**input_bit)

        # ラベル判別用の回路を、入力に追加で設定
        input_circuit = QuantumCircuit(input_bit + self.trash_bit_num)
        input_circuit = input_circuit.compose(feature_mapping, range(input_bit))
        
        # Rxゲートの設定が必要なビット数を取得
        for i in range(self.trash_bit_num):
            if 2**i <= label_num <= 2**(i+1):
                rx_trash_bit_num = i + 1
                break
        for i in range(rx_trash_bit_num):
            input_circuit.rx(Parameter("Label[{}]".format(i)), input_bit + i)
                

        # AE回路（入力～スワップテスト）を作成
        ae = self.auto_encoder_circuit()

        # 総量子ビットを用意して回路を作る
        train_qc = QuantumCircuit(input_bit + self.trash_bit_num + 1, 1)

        # 回路を登録
        train_qc = train_qc.compose(input_circuit, range(self.latent_bit_num + self.trash_bit_num*2))
        train_qc = train_qc.compose(ae)
        
        return train_qc, input_circuit.parameters, ae.parameters
        

    def classification_circuit(self):
        input_bit = self.latent_bit_num + self.trash_bit_num
        
        test_qr = QuantumRegister(input_bit, "q")
        test_cr = ClassicalRegister(self.trash_bit_num, "c")
        
        test_qc = QuantumCircuit(test_qr, test_cr)
        
        feature_mapping = RawFeatureVector(2**input_bit)
        test_qc = test_qc.compose(feature_mapping)
        
        test_qc = test_qc.compose(Ansatz(self.ansatz_dict, self.latent_bit_num, self.trash_bit_num).ansatz)
        
        for i in range(self.trash_bit_num):
            test_qc.measure(self.latent_bit_num + i, test_cr[i])
            
        return test_qc