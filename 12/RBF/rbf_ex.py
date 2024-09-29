import numpy as np
from scipy.interpolate import Rbf
from matplotlib import pyplot as plt
def sample_func_g(x):
    # return x + 20*np.sin(2*np.pi*x/60) + 100*np.exp(-((x-50)/8)**2)
    return x**2
x_j = np.arange(-95, 95,10)
y_j = sample_func_g(x_j) 

x = np.arange(-101, 101,1)
y = sample_func_g(x)
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
ax.plot(x, y, color='magenta', lw=3, label=r'$g(x)$')
ax.scatter(x_j, y_j, color='blue', s=100, zorder=2, label=r'$x_j$')
ax.grid()
ax.set_xlim(-100,100)
ax.set_ylim(0,140)
ax.legend(fontsize=16)
plt.savefig('rbf_ex.png')
plt.close()

#補完した結果
function_list=['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate']
fig, axes = plt.subplots(figsize=(32,10) ,nrows=2, ncols=4)
for i, function in enumerate(function_list):
    row_num = i//axes.shape[1]
    col_num = i%axes.shape[1]
    ax = axes[row_num, col_num]

    interp_model = Rbf(x_j, y_j, function=function)

    ax.scatter(x_j, y_j, color='blue', s=50, zorder=2, label='points for interpolation')
    ax.plot(x, interp_model(x), color='green', linestyle='dashed', lw=3, label=r'$f(x)$')
    ax.plot(x,y, color='magenta', lw=3, label=r'$g(x)$')
    ax.set_title(function, fontsize=16)
    ax.grid()
    if i == 0:
        ax.legend(fontsize=12)
    ax.set_xlim(-100,100)
    ax.set_ylim(0,140)
# 画像を保存
plt.tight_layout()  # プロットが重ならないように調整
plt.savefig('interpolation_results.png', dpi=300, bbox_inches='tight')  # 保存するファイル名を指定
