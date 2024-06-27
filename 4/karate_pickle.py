import torch
import torch_geometric.datasets as datasets
import pickle

# データをロード
data = datasets.KarateClub()

# pickle形式で保存
with open('karate.pkl', 'wb') as f:
    pickle.dump(data, f)
