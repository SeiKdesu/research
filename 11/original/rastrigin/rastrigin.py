import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.pyplot as plt
import numpy as np

def rastrigin(*X, **kwargs):
    A = kwargs.get('A', 10)
    return A + sum([(x**2 - A * np.cos(2 * math.pi * x)) for x in X])


# xとyの範囲を設定
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

# メッシュグリッドの作成
X, Y = np.meshgrid(x, y)

# Z値の計算 (ここではシンプルな関数を使用)
X, Y = np.meshgrid(X, Y)

Z = rastrigin(X, Y, A=10)

print(Z.shape)
# fig = plt.figure()
# ax = fig.gca()

# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.plasma, linewidth=0, antialiased=False)    
# plt.savefig('rastrigin.png')
# plt.close()

# 等高線を描画
plt.contour(X, Y, Z, levels=10, cmap='viridis')

# カラーバーの追加
plt.colorbar()

# グラフの表示
plt.savefig('rastrigin_contour.png')
plt.close()

# 塗りつぶされた等高線を描画
plt.contourf(X, Y, Z, levels=10, cmap='viridis')

# カラーバーの追加
plt.colorbar()

# グラフの表示
plt.savefig('rastrigin_contour_color.png')
plt.close()


# 塗りつぶされた等高線を描画
plt.contourf(X, Y, Z, levels=10, cmap='viridis')

# カラーバーの追加
plt.colorbar()

# グラフの表示
plt.savefig('rastrigin_contour_num.png')
plt.close()

