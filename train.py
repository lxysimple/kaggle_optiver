import pandas as pd
import lightgbm as lgb 
import xgboost as xgb 
import catboost as cbt 
import numpy as np 
import joblib 
import os 
import copy

# 控制代码流程的超参数
TRAINING = True
# TRAINING = False # 这个时候只用数据增强函数+15个模型，进行推理

if TRAINING:
    # 上传到kaggle dataset上的压缩文件会自动解压，厉害！ 
    df_ = pd.read_csv('train_end.csv')


# 检查目录是否存在
directory = "models"
if not os.path.exists(directory):
    # 如果不存在，创建目录
    os.mkdir(directory)

N_fold = 5 

if TRAINING:
    Y = df_['target'].values
    X = df_.drop(columns='target')

    # 去除标签是NAN的行
    X = X[np.isfinite(Y)]
    Y = Y[np.isfinite(Y)]

    index = np.arange(len(X))


models = {'lgb':[],'xgb':[],'cbt':[]}

def train(model_dict, modelname='lgb', i=0, model_path= '/kaggle/input/optiverbaselinezyz'):
    

    if TRAINING:
        model = copy.deepcopy(model_dict[modelname])
        
        model.fit( 
                    X[index%N_fold!=i], Y[index%N_fold!=i], # 用这种算法实现5折交叉验证，很巧妙
                    eval_set=[(X[index%N_fold==i], Y[index%N_fold==i])],
                    verbose=50, # 每50个迭代输出一次log
                    # 在连续100个迭代内如果loss不再改善，就会停止 
                    early_stopping_rounds=100
        )
        # 将该折模型保存到内存和磁盘，太占内存了，训练结束后取消训练模式再取模型吧
#         models.append(model)
        joblib.dump(model, f'./models/{modelname}_{i}.model') 
        
        # 删除深拷贝对象
        del model

    else:
        # 依次将15个模型取出，用于之后的推理
        models[modelname].append(joblib.load(f'{model_path}/{modelname}_{i}.model'))
    return 

model_dict = {
    # L1损失函数、500棵树=迭代500次（每次迭代就是基于train数据构建一棵树然后拟合残差）
    'lgb': lgb.LGBMRegressor(
        objective='regression_l1', 
        n_estimators=500, device= 'gpu',
        gpu_device_id = 1
    ),
    # 使用基于直方图的方法来构建决策树、L1损失、500棵树
    'xgb': xgb.XGBRegressor(
        booster = 'gbtree',
        tree_method='gpu_hist', 
        objective='reg:absoluteerror', 
        eval_metric = 'mae',
        n_estimators=500,
        learning_rate = 0.02,
        alpha = 8,
        max_depth = 4,
        subsample = 0.8,
        colsample_bytree = 0.5,
        seed = 42
    ),
    # 最小化平均绝对误差、迭代训练3000次
    'cbt': cbt.CatBoostRegressor(objective='MAE', iterations=3000, task_type= 'GPU')
}


# lgb_0：6.34879
# lgb_1：6.20357
# lgb_2：6.15322
# lgb_3：6.22028
# lgb_4：6.34358
for i in range(N_fold): 
    train(model_dict, 'lgb', i, '/kaggle/input/optiver-v1')






