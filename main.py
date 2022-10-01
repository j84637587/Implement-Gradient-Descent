from math import e, sqrt


def argminf(x1: float, x2: float) -> float:
    """帶入值到方程式

    Args:
        x1 (float): 第一個未知數的值
        x2 (float): 第二個未知數的值

    Returns:
        float: 計算結果
    """
    r = -200 * (e ** (-0.02 * sqrt(x1**2+x2**2)))
    return r


def deriv(x1: float, x2: float) -> tuple[float, float]:
    """各別對每個未知數, 帶值計算一階偏微分.

    Args:
        x1 (float): 第一個參數.
        x2 (float): 第二個參數.

    Returns:
        tuple[float, float]: 計算結果.
    """
    rf = argminf(x1, x2)
    deriv_x1 = rf * ((-0.02 * x1) / sqrt(x1**2+x2**2)) # 對 x1 做偏微分
    deriv_x2 = rf * ((-0.02 * x2) / sqrt(x1**2+x2**2)) # 對 x2 做偏微分
    return deriv_x1, deriv_x2


def gradient_decs() -> tuple[float, float, float]:
    """對函數做梯度下降

    Returns:
        tuple[float, float, float]: 最後迭代收斂結果
    """

    alpha = 0.01            # 學習率
    convergenc = 1e-6       # 收斂值
    (x1, x2) = (-32, -32)   # 初始值 注意: 不要初始化 0, 0 因為偏微分分母會是0

    y1 = argminf(x1, x2)
    while True:
        deriv_x1, deriv_x2 = deriv(x1, x2)
        x1 = x1 - alpha * deriv_x1
        x2 = x2 - alpha * deriv_x2
        y2 = argminf(x1, x2)
        if y1 - y2 < convergenc:
            # 如果收斂了就返回
            return x1, x2, y2
        if y2 < y1:
            # 更新當前最小值
            y1 = y2


if __name__ == '__main__':
    result = gradient_decs()
    print(result)
