import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import polars as pl
from tqdm import tqdm


NUMS = [ 'imbalance_size',
       'imbalance_buy_sell_flag', 'reference_price', 'matched_size',
       'far_price', 'near_price', 'bid_price', 'bid_size', 'ask_price',
       'ask_size', 'wap','bid_ask_imb_minus', 'imbalance_match_imb_minus']

prices = ['reference_price','far_price', 'near_price', 'ask_price', 'bid_price', 'wap']

gaps = [i for i in range(1, 56)]


columns_convert = [pl.col(i).cast(pl.Float32) for i in NUMS if i not in ('bid_ask_imb_minus', 'imbalance_match_imb_minus')] 
columns_convert += [(((pl.col('bid_size') - pl.col('ask_size')) ).alias(f'bid_ask_imb_minus')),
                    (((pl.col('imbalance_size') - pl.col('matched_size')) ).alias(f'imbalance_match_imb_minus')),
                   
                   ]
columns = [] 

for i,a in enumerate(prices):
    for j,b in enumerate(prices):
        if i>j:
            columns.append( (((pl.col(a) - pl.col(b))/(pl.col(a)+pl.col(b)) ).alias(f'{a}_{b}_imb')) )
            columns.append( (((pl.col(a) - pl.col(b)) ).alias(f'{a}_{b}_imb_minus')) )
            
columns_part2 = [ 
    ((pl.col("imbalance_size") - pl.col("imbalance_size").shift(1)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("imbalance_size_diff_0")),
    ((pl.col("imbalance_size") - pl.col("imbalance_size").shift(2)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("imbalance_size_diff_1")),
    ((pl.col("imbalance_size") - pl.col("imbalance_size").shift(3)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("imbalance_size_diff_2")),
    ((pl.col("imbalance_size") - pl.col("imbalance_size").shift(4)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("imbalance_size_diff_3")),
    ((pl.col("imbalance_size") - pl.col("imbalance_size").shift(5)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("imbalance_size_diff_4")),
    ((pl.col("imbalance_size") - pl.col("imbalance_size").shift(6)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("imbalance_size_diff_5")),

    ((pl.col("matched_size") - pl.col("matched_size").shift(1)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("matched_size_diff_0")),
    ((pl.col("matched_size") - pl.col("matched_size").shift(2)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("matched_size_diff_1")),
    ((pl.col("matched_size") - pl.col("matched_size").shift(3)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("matched_size_diff_2")),
    ((pl.col("matched_size") - pl.col("matched_size").shift(4)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("matched_size_diff_3")),
    ((pl.col("matched_size") - pl.col("matched_size").shift(5)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("matched_size_diff_4")),
    ((pl.col("matched_size") - pl.col("matched_size").shift(6)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("matched_size_diff_5")),
    
    ((pl.col("bid_size") - pl.col("bid_size").shift(1)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("bid_size_diff_0")),
    ((pl.col("bid_size") - pl.col("bid_size").shift(2)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("bid_size_diff_1")),
    ((pl.col("bid_size") - pl.col("bid_size").shift(3)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("bid_size_diff_2")),
    ((pl.col("bid_size") - pl.col("bid_size").shift(4)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("bid_size_diff_3")),
    ((pl.col("bid_size") - pl.col("bid_size").shift(5)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("bid_size_diff_4")),
    ((pl.col("bid_size") - pl.col("bid_size").shift(6)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("bid_size_diff_5")),

    ((pl.col("ask_size") - pl.col("ask_size").shift(1)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("ask_size_diff_0")),
    ((pl.col("ask_size") - pl.col("ask_size").shift(2)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("ask_size_diff_1")),
    ((pl.col("ask_size") - pl.col("ask_size").shift(3)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("ask_size_diff_2")),
    ((pl.col("ask_size") - pl.col("ask_size").shift(4)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("ask_size_diff_3")),
    ((pl.col("ask_size") - pl.col("ask_size").shift(5)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("ask_size_diff_4")),
    ((pl.col("ask_size") - pl.col("ask_size").shift(6)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("ask_size_diff_5")),
    
    ((pl.col("bid_ask_imb_minus") - pl.col("bid_ask_imb_minus").shift(1)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("bid_ask_imb_minus_diff_0")),
    ((pl.col("bid_ask_imb_minus") - pl.col("bid_ask_imb_minus").shift(2)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("bid_ask_imb_minus_diff_1")),
    ((pl.col("bid_ask_imb_minus") - pl.col("bid_ask_imb_minus").shift(3)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("bid_ask_imb_minus_diff_2")),
    ((pl.col("bid_ask_imb_minus") - pl.col("bid_ask_imb_minus").shift(4)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("bid_ask_imb_minus_diff_3")),
    ((pl.col("bid_ask_imb_minus") - pl.col("bid_ask_imb_minus").shift(5)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("bid_ask_imb_minus_diff_4")),
    ((pl.col("bid_ask_imb_minus") - pl.col("bid_ask_imb_minus").shift(6)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("bid_ask_imb_minus_diff_5")),
    
    ((pl.col("imbalance_match_imb_minus") - pl.col("imbalance_match_imb_minus").shift(1)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("imbalance_match_imb_minus_diff_0")),
    ((pl.col("imbalance_match_imb_minus") - pl.col("imbalance_match_imb_minus").shift(2)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("imbalance_match_imb_minus_diff_1")),
    ((pl.col("imbalance_match_imb_minus") - pl.col("imbalance_match_imb_minus").shift(3)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("imbalance_match_imb_minus_diff_2")),
    ((pl.col("imbalance_match_imb_minus") - pl.col("imbalance_match_imb_minus").shift(4)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("imbalance_match_imb_minus_diff_3")),
    ((pl.col("imbalance_match_imb_minus") - pl.col("imbalance_match_imb_minus").shift(5)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("imbalance_match_imb_minus_diff_4")),
    ((pl.col("imbalance_match_imb_minus") - pl.col("imbalance_match_imb_minus").shift(6)).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias("imbalance_match_imb_minus_diff_5")),
    
    ( (  (pl.col("matched_size") - pl.col("imbalance_size")) / (pl.col("matched_size") + pl.col("imbalance_size"))  ).alias("matched_imblance_diff_0") ),
    ( (  (pl.col("matched_size") - pl.col("imbalance_size"))   ).alias("matched_imblance_diff_1") ),
    # bid_size-ask_size
    ( (pl.col("bid_size") - pl.col("ask_size")).alias("bid_ask_diff_0") ),
    ( (  (pl.col("bid_size") - pl.col("ask_size")) / (pl.col("bid_size") + pl.col("ask_size"))  ).alias("bid_ask_diff_1") ),
]


mins = 60
day = 1
year = (365.2425)*day

columns_part3 = [
    ( np.sin((pl.col("seconds_in_bucket")) * (2 * np.pi / 1)).alias("second_sin") ),
    ( np.cos((pl.col("seconds_in_bucket") ) * (2 * np.pi / 1)).alias("second_cos") ),
]

columns_part4 = []
for num in NUMS:
    for gap in gaps:
        columns_part4.append(
            pl.col(num).shift(gap).fill_null(0).clip(-1e9, 1e9).over(["stock_id", "date_id"]).alias(f"{num}_gap_{gap}")
        )

columns += columns_part2 + columns_part4 + columns_part3

df = (pl.read_csv("train.csv") # 缩小数据量看能不能跑通
      .with_columns(columns_convert)
      .with_columns(columns)
      .drop(['row_id', 'time_id'])
     )


df.filter((pl.col('stock_id')==0))

df.filter((pl.col('stock_id')==0)&(pl.col('date_id')==0)).select(['imbalance_size', 'imbalance_size_diff_0'])

NUMS += [i for i in df.columns if i not in ['stock_id', 'date_id', 'seconds_in_bucket'] and i not in NUMS]


eps = 1e-3
def feature_engineer_for_index(x, date, time):
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
    
    return df


# train = []
# for date in tqdm(range(481)):
#     for t in range(0, 550, 10):
#         train.append(feature_engineer_for_index(df, date, t))

train = []
for date in tqdm(range(5)):
    for t in range(0, 5, 10):
        train.append(feature_engineer_for_index(df, date, t))


train = pl.concat(train)
train = train.to_pandas()
train.dropna(subset=['target'], inplace=True)

# some cleaning...
null = train.isnull().sum().sort_values(ascending=False) / len(train)


drop = list(null[null>0.9].index)


for col in train.columns:
    if train[col].nunique()==1:
        drop.append(col)
print("*********df DONE*********")

FEATURES = [c for c in train.columns if c not in drop + ['target', 'date_id']] 

print(len(FEATURES))

df.write_csv('train_end.csv')
