import matplotlib.pyplot as plt
import numpy as np
import math

def weight_histogram(coefficient, name, mode='normal', title='Feature'):
    coefficient = coefficient.copy()
    name = name.copy()

    if mode == 'cleanup': # remove zero coefficients
        new_coeff = []
        new_name = []
        for idx in range(len(coefficient)):
            if math.fabs(coefficient[idx]) > 1e-3:
                new_coeff.append(coefficient[idx])
                new_name.append(name[idx])

        coefficient = new_coeff
        name = new_name

    N = len(name)
    index = np.arange(N)
    # 柱子的宽度
    width = 0.3
    # 绘制柱状图, 每根柱子的颜色为紫罗兰色
    plt.bar(index, coefficient, width, label="value", color="#87CEFA")
    # 设置横轴标签
    plt.xlabel('Features')
    # 设置纵轴标签
    plt.ylabel('Weight')
    # 添加标题
    plt.title(title)
    # 添加纵横轴的刻度
    plt.xticks(index, name)
    plt.legend(loc="upper right")
    plt.show()
    return coefficient, name
