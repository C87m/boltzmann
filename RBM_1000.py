import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.datasets import load_digits

# 手書き数字を取得 scikit-learnから利用できるMNIST
digits = load_digits()

# 表示
def show_img(row, col, img_list1, img_list2, title_list1, title_list2, subtitle, subtitlesize, figsize):
    fig, ax = plt.subplots(row, col, figsize=figsize)
    fig.suptitle(subtitle, fontsize=subtitlesize, color='black')
    
    for i in range(col):
        if row == 1:
            img1 = np.reshape(img_list1[i], (8, 8))
            ax[i].imshow(img1, cmap='Greys')
            ax[i].set_title(title_list1[i])
        else:
            img1 = np.reshape(img_list1[i], (8, 8))
            ax[0, i].imshow(img1, cmap='Greys')
            ax[0, i].set_title(title_list1[i])
            
            img2 = np.reshape(img_list2[i], (8, 8))
            ax[1, i].imshow(img2, cmap='Greys')
            ax[1, i].set_title(title_list2[i])
            
    plt.show()
    
# 0 のみのデータセットを取得
zero_index_list = [i for i, x in enumerate(digits.target) if x == 0]
raw_data_list = [digits.data[i] for i in zero_index_list]

num_data = 50 # 使用するデータの数
num_spin = len(raw_data_list[0]) #画像1枚のスピンの数

# データの加工
edit_data_list = []
for n in range(num_data):
    edit_data = [1 if raw_data_list[n][spin] >= 4 else 0 for spin in range(num_spin)]
    edit_data_list.append(edit_data)

class RBM:
    def __init__(self, n_visible, n_hidden, learning_rate=0.1, n_iterations=1000):
        self.n_visible = n_visible #可視ユニットの数
        self.n_hidden = n_hidden #隠れユニットの数
        self.learning_rate = learning_rate #学習率
        self.n_iterations = n_iterations #反復回数

        self.weights = np.random.randn(n_visible, n_hidden) * 0.01 #重み行列。小さなランダム値で初期化
        self.visible_bias = np.zeros(n_visible) #可視ユニットのバイアス
        self.hidden_bias = np.zeros(n_hidden) #隠れユニットのバイアス

    def sigmoid(self, x): #シグモイド関数、値を0から1にスケーリング
        return 1 / (1 + np.exp(-x))

    def train(self, data):
        for iteration in range(self.n_iterations):
            # 正のフェーズ
            #可視データから隠れユニットの活性化確率を計算
            pos_hidden_probs = self.sigmoid(np.dot(data, self.weights) + self.hidden_bias)
            #活性化確率を使って隠れユニットの状態をサンプリング
            pos_hidden_states = (pos_hidden_probs > np.random.rand(len(data), self.n_hidden)).astype(float)
            #可視ユニットと隠れユニットの共起を計算
            pos_associations = np.dot(data.T, pos_hidden_probs)

            # 負のフェーズ
            #隠れユニットの状態から再構成された可視データの確率を計算
            neg_visible_probs = self.sigmoid(np.dot(pos_hidden_states, self.weights.T) + self.visible_bias)
            #バイナリ状態に変換
            neg_visible_probs = (neg_visible_probs > np.random.rand(len(data), self.n_visible)).astype(float)
            #再構成された可視データから隠れユニットの活性化確率を計算
            neg_hidden_probs = self.sigmoid(np.dot(neg_visible_probs, self.weights) + self.hidden_bias)

            # 重みとバイアスの更新
            #再構成されたデータの可視ユニットと隠れユニットの共起を計算
            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

            #重みとバイアスを正負のフェーズの差異に基づいて更新
            self.weights += self.learning_rate * ((pos_associations - neg_associations) / len(data))
            self.visible_bias += self.learning_rate * np.mean(data - neg_visible_probs, axis=0)
            self.hidden_bias += self.learning_rate * np.mean(pos_hidden_probs - neg_hidden_probs, axis=0)

            # エラーログ
            #100回のイテレーションごとに元のデータと再構成データとの誤差を計算して表示
            if iteration % 100 == 0:
                error = np.mean((data - neg_visible_probs) ** 2)
                print(f"Iteration: {iteration}, Error: {error}")

    #可視層から隠れ層への変換
    #可視ユニットのデータを入力として、隠れユニットの状態をサンプリングして返す
    def run_visible(self, data):
        hidden_probs = self.sigmoid(np.dot(data, self.weights) + self.hidden_bias)
        hidden_states = (hidden_probs > np.random.rand(len(data), self.n_hidden)).astype(float)
        return hidden_states

    #隠れ層から可視層への変換
    #隠れユニットのデータを入力として、可視ユニットの状態をサンプリングして返す
    def run_hidden(self, hidden_data):
        visible_probs = self.sigmoid(np.dot(hidden_data, self.weights.T) + self.visible_bias)
        visible_states = (visible_probs > np.random.rand(len(hidden_data), self.n_visible)).astype(float)
        return visible_states

    #データの再構成
    #入力データを隠れユニットを通して再構成し、再構成されたデータを返す
    def reconstruct(self, data):
        hidden_states = self.run_visible(data)
        reconstructed_data = self.run_hidden(hidden_states)
        return reconstructed_data

# サンプルデータ
data = np.array(edit_data_list)

# RBMの初期化とトレーニング
rbm = RBM(n_visible=64, n_hidden=20, learning_rate=0.1, n_iterations=1000)
rbm.train(data)

# 再構成のテスト
sample_data = np.array(data[1])
reconstructed_data = rbm.reconstruct(sample_data)

redata = reconstructed_data[0]

print("Original Data:", data[1])
print("Reconstructed Data:", redata)
    
list1 = []
list2 = []
list1.append(data[1])
list1.append(redata)
show_img(row=1, col=2, img_list1=list1, img_list2=None,
         title_list1="sample_reconstructed", title_list2=None,
         subtitle="", subtitlesize=24, figsize=(14, 3))