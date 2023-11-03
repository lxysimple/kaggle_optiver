import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import polars as pl
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
from tqdm import tqdm


train = pd.read_csv('train_end.csv')

train_df = train[train['date_id']< 450]
val_df = train[train['date_id'] >= 450 ]



train_x, train_y = train_df[FEATURES], train_df[['target']]
valid_x, valid_y = val_df[FEATURES], val_df[['target']]


# 模型配置
xgb_params = {
        'booster': 'gbtree',
        'tree_method': 'gpu_hist',
        'objective': 'reg:squarederror',
        'eval_metric':'mae',
        'learning_rate': 0.02,
        'alpha': 8,
        'max_depth': 4,
        'subsample':0.8,
        'colsample_bytree': 0.5,
        'seed': 42
        }
xgb_params['n_estimators'] = 500
lgbm_params = {
    'objective' : 'regression_l1',
    'num_iterations': 500,
 
}


















