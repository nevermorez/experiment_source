import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def plt_scatter(y_test, y_pred,jpg_name):
    plt.clf()
    plt.scatter(y_test, y_pred, color='red')
# plt.scatter(y_test_filter, y_pred_filter,color='red')
    plt.xlim(115,200)
    plt.ylim(115,200)
# 设置标题和坐标轴标签
    plt.title('Real vs Predicted Values')
    plt.xlabel('Real Values')
    plt.ylabel('Predicted Values')

    # 可选：绘制45度参考线
    plt.plot([115,200],[115,200], color='red', linestyle='--')
    plt.savefig(jpg_name + '.jpg')



def combine_fea(input_data,mode):
	#分别以乘，除，混合的方式构造数据
    res = np.zeros((input_data.shape[0], 60))
    if mode == 'mutil':
        for i in range(0,120,2):
            res[:,i//2] = input_data[:,i] * (input_data[:,i+1] + 1e-10)
        res_total = np.concatenate((input_data, res), axis=1)
        return res_total,res
    elif mode == 'division':
        for i in range(0,120,2):
            res[:,i//2] = input_data[:,i] / (input_data[:,i+1] + 1e-10)
        res_total = np.concatenate((input_data, res), axis=1)
        return res_total,res
    elif mode == 'mixed':
        res_mutil = np.zeros((input_data.shape[0], 60))
        res_division = np.zeros((input_data.shape[0], 60))
        for i in range(0,120,2):
            res_mutil[:,i//2] = input_data[:,i] * (input_data[:,i+1] + 1e-10)
            res_division[:,i//2] = input_data[:,i] / (input_data[:,i+1] + 1e-10)
        res_total = np.concatenate((input_data, res_mutil,res_division), axis=1)
        res_temp = np.concatenate((res_mutil,res_division), axis=1)
        return res_total,res_temp
