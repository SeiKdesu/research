import os
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx
import torch.nn.functional as F
from datetime import datetime
from tutorial_rbf import QOL, objective_function, objective_function1

# GPU設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 結果保存用のディレクトリ設定
def setup_directories():
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = f'results/run_{current_time}'
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

# グラフ可視化関数
def visualize_graph(G, color, result_dir, filename):
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    
    # ノードの配置設定
    pos = {}
    ranges = [[0, 1, 2, 3, 4, 5], 
              [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], 
              [27]]
    x_offset = 0
    
    for r in ranges:
        for i, node in enumerate(r):
            pos[node] = (x_offset, -i)
        x_offset += 1
    
    # グラフ描画
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=color, cmap=plt.cm.rainbow)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
    
    plt.savefig(f'{result_dir}/{filename}.png')
    plt.close()

# 改善されたGNNモデル
class ImprovedGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        super(ImprovedGNN, self).__init__()
        
        # Convolution layers
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        
        # Fully connected layers
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = torch.nn.Linear(hidden_size // 2, num_classes)
        
        # Regularization
        self.dropout = torch.nn.Dropout(0.3)
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_size)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First convolution block
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second convolution block
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

# トレーニング関数
def train_model(model, dataset, optimizer, result_dir, max_epochs=500):
    best_loss = float('inf')
    best_state = None
    patience = 15
    patience_counter = 0
    losses = []
    accuracies = []
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(dataset)
        loss = F.nll_loss(out, dataset.y)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            pred = out.max(1)[1]
            correct = pred[:6].eq(dataset.y[:6]).sum().item()  # First 6 nodes only
            acc = correct / 6
            
            losses.append(loss.item())
            accuracies.append(acc)
        
        # Learning rate scheduling
        scheduler.step(loss)
        
        # Early stopping check
        if loss < best_loss:
            best_loss = loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
            
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
            
            # Optimization results for visualization
            out = model(dataset)
            pred = out.max(1)[1]
            
            # Get predictions and calculate objective function
            pp, _ = QOL()
            ff_out = objective_function(pp, 6)  # Assuming dim=6
            
            mal_list0 = [i for i in range(6) if pred[i] == 0]
            mal_list1 = [i for i in range(6) if pred[i] == 1]
            
            all_prediction = objective_function1(pp, np.array(mal_list0), np.array(mal_list1))
            
            print(f'Teacher output: {ff_out[0]}, Predicted output: {all_prediction[0]}')
            
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(f'{result_dir}/training_curves.png')
    plt.close()
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, best_loss

def main():
    # ディレクトリ設定
    result_dir = setup_directories()
    
    # グラフデータの作成
    src = []
    dst = []
    
    # Source nodes
    for j in range(6):
        for i in range(21):
            src.append(j)
    for i in range(6, 27):
        src.append(i)
        
    # Destination nodes
    for j in range(6):
        for i in range(6, 27):
            dst.append(i)
    for i in range(21):
        dst.append(27)
        
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # ノード特徴量の作成
    pp, _ = QOL()
    formatted_weight_data = []
    
    # First 6 nodes (input nodes)
    for i in range(5, -1, -1):
        formatted_weight_data.append([pp[i], 1, 1, 1, 1, 1])
    
    # Middle nodes
    for _ in range(21):
        formatted_weight_data.append([1, 1, 1, 1, 1, 1])
    
    # Last node (output node)
    num1 = objective_function(pp, 6)
    formatted_weight_data.append([num1[0], 1, 1, 1, 1, 1])
    
    x = torch.tensor(formatted_weight_data, dtype=torch.float)
    
    # ラベルの作成
    y = torch.tensor([0]*3 + [1]*3 + [2]*22)
    
    # データセットの作成
    dataset = Data(x=x, edge_index=edge_index, y=y)
    dataset = dataset.to(device)
    
    # 初期グラフの可視化
    G = to_networkx(dataset, to_undirected=False)
    visualize_graph(G, dataset.y.cpu(), result_dir, 'initial_graph')
    
    # モデルの初期化
    model = ImprovedGNN(
        num_features=dataset.num_node_features,
        hidden_size=64,
        num_classes=3
    ).to(device)
    
    # オプティマイザの設定
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=5e-4
    )
    
    # モデルのトレーニング
    model, final_loss = train_model(model, dataset, optimizer, result_dir)
    
    # 最終結果の評価
    model.eval()
    with torch.no_grad():
        out = model(dataset)
        pred = out.max(1)[1]
        acc = pred[:6].eq(dataset.y[:6]).sum().item() / 6
        
        print(f'\nFinal Results:')
        print(f'Loss: {final_loss:.4f}')
        print(f'Accuracy: {acc:.4f}')
        print(f'Predictions: {pred[:6].cpu().numpy()}')
        print(f'True labels: {dataset.y[:6].cpu().numpy()}')
        
        # 最終グラフの可視化
        visualize_graph(G, pred.cpu(), result_dir, 'final_graph')

if __name__ == "__main__":
    main()