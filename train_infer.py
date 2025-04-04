import pandas as pd
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 如果有显卡就用显卡训练，否则用cpu
from model import PairwiseCNN
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
import csv
import os
from utils import plt_scatter,combine_fea


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

# CSV文件路径
file_path = ''  # 替换为你的CSV文件路径
file_path2 = ''
file_path3 = ''
res_dir = 'result_cnn/experimenau_ori'
model_dir = 'cnn_model/experimenau_ori'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)  
if not os.path.exists(res_dir + '/pic'):
    os.makedirs(res_dir + '/pic')
if not os.path.exists(res_dir + '/res'):
    os.makedirs(res_dir + '/res')


# 使用Pandas读取CSV文件
# 假设您的CSV文件没有标题行，并且使用逗号分隔   
data3_df = pd.read_csv(file_path, delimiter='\t', skiprows=[0], header=None)

# 转换Pandas DataFrame为NumPy数组
data3_np = data3_df.values


input_data3 = data3_np[:, 1:121]
labels_3 = data3_np[:, 121]

#构造数据，可按实验条件添加至训练数据特征中
_,single_mix = combine_fea(input_data3,mode='mutil')

data_reshaped3 = input_data3.reshape(len(input_data3), 60, 2)

# 选择奇数列和偶数列
odd_columns3 = data_reshaped3[:, :60, 0]
even_columns3 = data_reshaped3[:, :60, 1]

# 重新排列数据，将奇数列放在前面，偶数列放在后面
rearranged_data3 = np.concatenate([odd_columns3, even_columns3], axis=1)
rearranged_data_2d3 = rearranged_data3.reshape(len(input_data3), 120)


scaler_s3 = StandardScaler()

scaler_s3.fit(rearranged_data_2d3)

X_s3= scaler_s3.transform(rearranged_data_2d3)

data_reshaped3 = X_s3.reshape(len(input_data3), 2, 60)

input_data3 = torch.tensor(data_reshaped3, dtype=torch.float32)
labels3 = torch.tensor(labels_3, dtype=torch.float32)


global_min_val_loss = []
for i in range(20):
    
    x_train, x_val, y_train, y_val = train_test_split(input_data3, labels3, test_size=0.15, random_state=i)
    loss_fn = nn.MSELoss() 

    x_train = x_train.to(device)
    x_val = x_val.to(device)
    y_train = y_train.to(device)
    y_val = y_val.to(device)

    model = PairwiseCNN()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001) # 定义优化器
    # scheduler = ExponentialLR(optimizer, gamma=0.95) # 动态调整学习率
    min_val_loss = 0.0
    min_val_10_list = []
    best_val = None
    best_val_list = []
    flag = True
    for epoch in range(25000): 
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        outputs = outputs.flatten()
        loss = loss_fn(outputs, y_train)
        loss.backward()
        optimizer.step()
        # scheduler.step()
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                predictions = model(x_val)
                predictions = predictions.flatten()
                eval_loss = loss_fn(predictions, y_val)
                if flag:
                    flag = False
                    min_val_loss = eval_loss.item()
                else:
                    min_val_loss = min(min_val_loss, eval_loss.item())
                    if eval_loss.item() == min_val_loss:
                        if len(min_val_10_list) < 10:
                            min_val_10_list.append(eval_loss.item())
                            best_val_list.append(predictions)
                        else:
                            if min_val_loss < min(min_val_10_list):
                                min_val_10_list.pop(min_val_10_list.index(max(min_val_10_list
                                
                            )))
                                best_val_list.pop(min_val_10_list.index(max(min_val_10_list
                                
                            )))
                                min_val_10_list.append(min_val_loss)
                                best_val_list.append(predictions)
                        best_val = predictions
                        torch.save(model.state_dict(), model_dir + "/best_metric"+str(i)+".pth") # 保存最终模型
                    else:
                        best_val = best_val
                print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
                print(f'Evaluation Loss: {eval_loss.item():.4f}')


    plt.clf()
    plt_scatter(y_val.cpu().numpy(), best_val.cpu().detach().numpy(),res_dir +'/pic/pic' + str(i))
    print(best_val)
    global_min_val_loss.append(min_val_loss)

    result_record_file = open(res_dir+"/res/res"+str(i)+".csv", 'w', newline='')
    result_writer = csv.writer(result_record_file)
    result_writer.writerow(["truth", "prediction"])
    for temp in range(len(y_val)):
        result_writer.writerow([y_val.tolist()[temp], best_val.tolist()[temp]])  # 将loss和学习率写入CSV文件

print("二十轮实验测试集的rmse为：",global_min_val_loss)
print("均值为：",np.mean(global_min_val_loss))
