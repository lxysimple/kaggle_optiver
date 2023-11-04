import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import polars as pl
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
from tqdm import tqdm


df = pl.read_csv("train_end.csv")

df = df[0:1000]


NUMS = [ 'imbalance_size',
       'imbalance_buy_sell_flag', 'reference_price', 'matched_size',
       'far_price', 'near_price', 'bid_price', 'bid_size', 'ask_price',
       'ask_size', 'wap','bid_ask_imb_minus', 'imbalance_match_imb_minus']

NUMS += [i for i in df.columns if i not in ['stock_id', 'date_id', 'seconds_in_bucket'] and i not in NUMS]


eps = 1e-3
def feature_engineer_for_index(x, date, time):
    #x = x.filter(((pl.col('seconds_in_bucket') <= time) & (pl.col('date_id') == date)) | (pl.col('date_id') < date) )
    x = x.filter(((pl.col('seconds_in_bucket') <= time) & (pl.col('date_id') == date)) )
    feature_suffix = 'part_1'
    aggs = [
        *[pl.col(c).quantile(0.1, "nearest").alias(f"{c}_quantile1_{feature_suffix}") for c in NUMS],
        *[pl.col(c).quantile(0.2, "nearest").alias(f"{c}_quantile2_{feature_suffix}") for c in NUMS],
        *[pl.col(c).quantile(0.3, "nearest").alias(f"{c}_quantile3_{feature_suffix}") for c in NUMS],
        *[pl.col(c).quantile(0.4, "nearest").alias(f"{c}_quantile4_{feature_suffix}") for c in NUMS],
        *[pl.col(c).quantile(0.5, "nearest").alias(f"{c}_quantile5_{feature_suffix}") for c in NUMS],
        *[pl.col(c).quantile(0.6, "nearest").alias(f"{c}_quantile6_{feature_suffix}") for c in NUMS],
        *[pl.col(c).quantile(0.7, "nearest").alias(f"{c}_quantile7_{feature_suffix}") for c in NUMS],
        *[pl.col(c).quantile(0.8, "nearest").alias(f"{c}_quantile8_{feature_suffix}") for c in NUMS],
        *[pl.col(c).quantile(0.9, "nearest").alias(f"{c}_quantile9_{feature_suffix}") for c in NUMS],
        *[pl.col(c).mean().alias(f"{c}_mean_{feature_suffix}") for c in NUMS],
        *[pl.col(c).std().alias(f"{c}_std_{feature_suffix}") for c in NUMS],
        *[pl.col(c).min().alias(f"{c}_min_{feature_suffix}") for c in NUMS],
        *[pl.col(c).max().alias(f"{c}_max_{feature_suffix}") for c in NUMS],
        *[pl.col(c).median().alias(f"{c}_median_{feature_suffix}") for c in NUMS],
        *[pl.col(c).sum().alias(f"{c}_sum_{feature_suffix}") for c in NUMS],
        *[pl.col(c).skew().alias(f"{c}_skew_{feature_suffix}") for c in NUMS],
        *[pl.col(c).kurtosis().alias(f"{c}_kurtosis_{feature_suffix}") for c in NUMS],
        # adding rank features
        *[pl.col(c).last().alias(f"{c}_rank_{feature_suffix}") for c in NUMS],
        ]

    features_part_1= x.groupby(["stock_id"], maintain_order=True).agg(aggs)
    
    x = x.filter((pl.col('seconds_in_bucket') == time) & (pl.col('date_id') == date))
    df = x.join(features_part_1, on='stock_id')
    # df = df.join(features_part_2, on=['stock_id', 'seconds_in_bucket'])
    # df = df.join(features_part_3, on=['seconds_in_bucket'])
    #df = df.filter((pl.col('seconds_in_bucket') == time) & (pl.col('date_id') == date))
    #df = df.to_pandas()
    
    return df


train = []
for date in tqdm(range(481)):
    for t in range(0, 550, 10):
        train.append(feature_engineer_for_index(df, date, t))
  

train = pl.concat(train)
train = train.to_pandas()
train.dropna(subset=['target'], inplace=True)

print(train.shape)

# # some cleaning...
# null = train.isnull().sum().sort_values(ascending=False) / len(train)
# drop = list(null[null>0.9].index)
# for col in train.columns:
#     if train[col].nunique()==1:
#         drop.append(col)

# # FEATURES = [c for c in train.columns if c not in drop + ['target', 'date_id']] 
# FEATURES = [c for c in train.columns if c not in drop + ['date_id']] # 注意这里将target放入了

# len(FEATURES)

train.to_csv('FEATURES_train.csv')





