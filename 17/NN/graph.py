import torch
from icecream import ic
from rosenbrock_nn import *
dim = 6


src=[]
dst=[]
for j in range(dim):
    for i in range(12):
        src.append(j)

for i in range(dim,12+dim):
    src.append(i)

for j in range(dim):
    for i in range(dim,12+dim):
        dst.append(i)
for i in range(12):
    dst.append(12+dim+1)
edge_index=torch.tensor([src,dst],dtype=torch.long)

unit_vectors = [[1 if i == j else 0 for i in range(dim)] for j in range(dim)]
for vector in unit_vectors:
    ic(vector)
ic(unit_vectors)
x = torch.tensor(unit_vectors,dtype=torch.float)
y_tmp=[]
for i in range(3):
    y_tmp.append(0)
for i in range(3,6):
    y_tmp.append(1)
# for i in range(3):
#     y_tmp.append(1)
for i in range(12):
    y_tmp.append(2)

y = torch.tensor(y_tmp)


a = weight_return()
ic(a[0].shape)
tmp = torch.tensor(a[0],dtype=torch.float)

ic(tmp)
fl_tmp = tmp.flatten()

ic(fl_tmp)

fl_tmp = fl_tmp.unsqueeze(0)

# 各要素をリストとして変換
formatted_tensor = [[torch.tensor([value]) for value in row] for row in fl_tmp]

# 結果の確認
for row in formatted_tensor:
    print(row)
formatted_tensor = torch.tensor(formatted_tensor)
ic(formatted_tensor.shape)