
def Rosenbrock(x, n):
    value = 0
    for i in range(n - 1):
        value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return value
def dixon_price(x,n):
    n = len(x)
    term1 = (x[0] - 1) ** 2
    term2 = sum([i * (2 * x[i]**2 - x[i-1])**2 for i in range(1, n)])
    return term1 + term2
# def booth(xy):
#     """
#     Booth関数を計算します。

#     引数:
#     xy : array-like
#         入力ベクトル [x, y]
    
#     戻り値:
#     float
#         Booth関数の値
#     """
#     x, y = xy[0], xy[1]
    
#     term1 = x + 2*y - 7
#     term2 = 2*x + y - 5
    
#     return term1**2 + term2**2

# def matyas_function(x):
#     return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]


def trid(x,n):
    """
    指定された次元数Dに対してTrid関数を計算します。

    引数:
    n : int
        次元数（変数の数）
    
    戻り値:
    float
        Trid関数の値
    """
    # 入力ベクトル x を生成（例として 1 から n までの整数）
   # x = np.arange(1, n + 1)
    
    # 最初の和の計算
    sum1 = 0
    for i in range(n):
        sum1 += (x[i] - 1)**2
    
    # 2つ目の和の計算（隣接する要素の積）
    sum2 = 0
    for i in range(1, n):
        sum2 += x[i] * x[i-1]
    
    return sum1 - sum2
def objective_function3(x,dim):
    n_rosenbrock = 50
    n_dixon=50
    n_powell=32
    rosen_value = Rosenbrock(x[:n_rosenbrock], n_rosenbrock)
    dixon_value = dixon_price(x[n_rosenbrock:n_rosenbrock+n_dixon],n_dixon)
    rosen_value2 = Rosenbrock(x[n_rosenbrock+n_dixon:n_rosenbrock+n_dixon+n_rosenbrock], n_rosenbrock)
    
    # powell_value= powell(x[n_rosenbrock+n_dixon:n_rosenbrock+n_dixon+n_powell],n_powell)
    return rosen_value + dixon_value+rosen_value2