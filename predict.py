import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from sklearn.metrics import mean_squared_error
import datetime as dt
from sklearn import preprocessing

test_mode = False

train = pd.read_csv(os.path.join('data', 'sales_train_evaluation.csv'))
# np.random.seed(0)
# items = train['item_id'].unique()
# item_sample = np.random.choice(items, 100, replace=False)
# train = train.loc[train['item_id'].isin(item_sample)]
calendar = pd.read_csv(os.path.join('data', 'calendar.csv'))
sell_prices = pd.read_csv(os.path.join('data', 'sell_prices.csv'))

train_cols = []
cat_cols = []

# date features
calendar['d'] = calendar['d'].str[2:].astype(np.int16)
calendar['datetime'] = pd.to_datetime(calendar['date'])
calendar['day_of_week'] = calendar['datetime'].dt.dayofweek.astype(np.int64)
calendar['month'] = calendar['datetime'].dt.month.astype(np.int64)
calendar['wm_yr_wk'] = calendar['wm_yr_wk'].astype(np.int64)

# encode the items
enc_dict = {}
for col in ['item_id', 'store_id', 'dept_id']:
    enc = preprocessing.LabelEncoder()
    original_labels = train[col].values.flatten()
    if col in sell_prices.columns:
        original_labels = np.concatenate([original_labels, sell_prices[col].values.flatten()])
    enc.fit(original_labels)
    train[col] = enc.transform(train[col]).astype(np.int16)
    if col in sell_prices.columns:
        sell_prices[col] = enc.transform(sell_prices[col]).astype(np.int16)
    enc_dict[col] = enc
train.drop(['id', 'cat_id'], inplace=True, axis=1)

# reshape the training data
new_df = pd.DataFrame(index=train.index, columns=['d_' + str(i) for i in range(1942, 1970)])
train = pd.concat([train, new_df], axis=1)
train = pd.melt(train, id_vars=['item_id', 'dept_id', 'store_id', 'state_id'])
train['value'] = train['value'].astype(np.float64)
train.rename(columns={'variable': 'd'}, inplace=True)
train['d'] = train['d'].str[2:].astype(np.int64)
train.set_index(['item_id', 'dept_id', 'store_id', 'd'], inplace=True, drop=True)

train['dept_id_col'] = train.index.get_level_values('dept_id').astype(np.int64)
train['store_id_col'] = train.index.get_level_values('store_id').astype(np.int64)
train_cols += ['dept_id_col', 'store_id_col']
cat_cols += ['dept_id_col', 'store_id_col']

# use only 3 years of data
d_min = 365 * 3 + 28
train.drop(train.index[train.index.get_level_values('d') < d_min], inplace=True)

# merge calendar data
calendar.set_index('d', inplace=True, drop=True)
train = train.join(calendar[['day_of_week',  'month', 'wm_yr_wk', 'snap_WI', 'snap_CA', 'snap_TX']], how='left')
train['snap'] = 0
train['snap'] = train['snap'].astype(np.int8)
for state_id in ['CA', 'TX', 'WI']:
    train['snap'] += (train['snap_' + state_id] == 1) & (train['state_id'] == state_id)
train.drop(['state_id', 'snap_WI', 'snap_CA', 'snap_TX'], inplace=True, axis=1)
train.sort_index(inplace=True)
train_cols += ['day_of_week', 'month', 'snap']

# calculate first sale day
first_sale_ind = train['value'].ne(0).groupby(['item_id', 'store_id']).idxmax().to_frame('ind')
first_sale_ind['first_day'] = train.loc[first_sale_ind['ind'].values].index.get_level_values('d')
train = train.join(first_sale_ind[['first_day']], how='left')
train['days_since_first_sale'] = train.index.get_level_values('d') - train['first_day']
train.loc[train['days_since_first_sale'] <= 0, 'value'] = np.nan
train_cols.append('days_since_first_sale')
train.drop('first_day', axis=1, inplace=True)
train.sort_index(inplace=True)
del first_sale_ind

# price features
sell_prices['wm_yr_wk'] = sell_prices['wm_yr_wk'].astype(np.int16)
train.set_index('wm_yr_wk', append=True, drop=True, inplace=True)
sell_prices.set_index(['item_id', 'store_id', 'wm_yr_wk'], inplace=True)
train = train.join(sell_prices, how='left')
train.index = train.index.droplevel('wm_yr_wk')
train.sort_index(inplace=True)
for n in [7, 14]:
    gb = train.groupby(['store_id', 'item_id'])['sell_price'].rolling(window=n).mean().shift(1)
    train['sell_price_lag_{}'.format(n)] = gb.values - train['sell_price']
    train_cols += ['sell_price_lag_{}'.format(n)]
train_cols += ['sell_price']
del sell_prices


def calc_aggs(train):

    new_train_cols = []

    # events
    for e in calendar.loc[calendar['event_name_1'].notnull(), 'event_name_1'].unique():
        for n in range(0, 14):
            calendar['event_{}_lag_{}'.format(e, n)] = ((calendar['event_name_1'] == e)
                                                        | (calendar['event_name_2'] == e)).astype(np.int8).shift(n)
    group = ['item_id']
    agg_col = '_'.join(group)
    temp_df = train.groupby(group + ['d'])['value'].mean().to_frame('sum')
    events = calendar.loc[calendar['event_name_1'].notnull(), 'event_name_1'].unique()
    cal_cols = ['event_{}_lag_{}'.format(e, n) for e in events for n in range(0, 14)]
    temp_df = pd.merge(temp_df, calendar[cal_cols], 'left', left_index=True, right_index=True)
    new_col = 'event_uplift_' + agg_col
    temp_df[new_col] = 0
    for e in events:
        print(e)
        for n in range(0, 7):
            uplift = temp_df.groupby('event_{}_lag_{}'.format(e, n))['sum'].transform('mean') \
                         - temp_df.groupby('event_{}_lag_{}'.format(e, n+7))['sum'].transform('mean')
            temp_df[new_col] += uplift.fillna(0)
    if new_col in train.columns:
        train.drop(new_col, axis=1, inplace=True)
    train = pd.merge(train, temp_df[[new_col]], 'left', left_index=True, right_index=True)
    train = train.reorder_levels(['item_id', 'store_id', 'd', 'dept_id'], axis=0)
    train.sort_index(inplace=True)
    new_train_cols.append(new_col)

    # cumulative mean, variance, and seasonality features
    for season in ['day_of_week']:
        for group in [['dept_id', 'store_id'], ['item_id'], ['item_id', 'store_id']]:
            agg_col = '_'.join(group)
            new_cols = [agg_col + '_' + season + '_var', agg_col + '_' + season + '_mean', agg_col + '_var',
                        agg_col + '_mean']
            train.drop(new_cols, axis=1, inplace=True, errors='ignore')
            group_mean = train.groupby(group + ['d', 'day_of_week'])[['value']].mean()
            gb = group_mean.groupby(group, group_keys=False, as_index=False)['value'].expanding().agg(['mean', 'var']).shift(1).fillna(method='ffill')
            gb.index = gb.index.droplevel(0)
            gb.index = gb.index.droplevel('day_of_week')
            group_mean[[agg_col + '_mean', agg_col + '_var']] = gb
            gb = group_mean[['value']].groupby(group + [season], group_keys=False, as_index=False).expanding().agg(['mean', 'var']).shift(1).fillna(method='ffill')
            gb.index = gb.index.droplevel(0)
            group_mean[[agg_col + '_' + season + '_mean', agg_col + '_' + season + '_var']] = gb
            group_mean.index = group_mean.index.droplevel('day_of_week')
            train = pd.merge(train, group_mean[new_cols], 'left', left_index=True, right_index=True)
            train = train.reorder_levels(['item_id', 'store_id', 'd', 'dept_id'], axis=0)
            train.sort_index(inplace=True)
            train[agg_col + '_' + season + '_mean'] /= train[agg_col + '_mean']
            train[agg_col + '_' + season + '_var'] /= train[agg_col + '_mean']
            new_train_cols += new_cols
    return new_train_cols, train

last_predict = 1941
first_predict = last_predict - 27
store_closed = train.groupby(['store_id', 'd'])['value'].transform('sum') == 0
train.loc[store_closed & (train.index.get_level_values('d') < first_predict), 'value'] = np.nan
train['value_copy'] = train['value'].copy()
train.loc[train.index.get_level_values('d') >= 1914, 'value'] = np.nan
new_train_cols, train = calc_aggs(train)
train['value'] = train['value_copy']
train_cols += new_train_cols

# remove unneeded data now that we have calculated the features
keep_cols = ['value', 'value_copy', 'days_since_first_sale']
valid_cols = [c for c in list(np.unique(train_cols + keep_cols)) if c not in train.index.names]
train = train[valid_cols]


def calc_lags(train, n_blank):
    train_copy = train.copy()
    train_cols = []
    groups = [['item_id', 'store_id'], ['dept_id', 'store_id'], ['item_id']]
    for group in groups:
        agg_col = '_'.join(group)
        group_mean = train_copy.groupby(group + ['d'])[['value']].mean()
        new_cols = []
        train_copy.drop(['last_{}_days_avg_{}'.format(n, agg_col) for n in [7, 14, 28]], axis=1, inplace=True, errors='ignore')
        for n in [7, 14, 28]:
            gb = group_mean[['value']].groupby(group, as_index=False)['value'].rolling(window=n).mean().shift(1 + n_blank)
            gb.index = gb.index.droplevel(0)
            group_mean['last_{}_days_avg_{}'.format(n, agg_col)] = gb
            new_cols.append('last_{}_days_avg_{}'.format(n, agg_col))
        train_copy.drop(['lag_{}_days_avg_{}'.format(n, agg_col) for n in range(1, 8)], axis=1, inplace=True, errors='ignore')
        for n in range(1, 8):
            group_mean['lag_{}_days_avg_{}'.format(n, agg_col)] = group_mean[['value']].groupby(group)['value'].shift(n + n_blank)
            new_cols.append('lag_{}_days_avg_{}'.format(n, agg_col))
        train_copy = pd.merge(train_copy, group_mean[new_cols], 'left', left_index=True, right_index=True)
        train_copy = train_copy.reorder_levels(['item_id', 'store_id', 'd', 'dept_id'], axis=0)
        train_copy.sort_index(inplace=True)
        train_cols += new_cols
    return train_cols, train_copy


# train['value_copy'] = train['value'].copy()
# train.loc[train.index.get_level_values('d') >= 1914, 'value'] = np.nan
new_train_cols, train = calc_lags(train, 0)
# train['value'] = train['value_copy']
train_cols += new_train_cols


def predict(first_day, last_day, train, est, num_rounds, use_test=True):
    train_copy = train.loc[(train.index.get_level_values('d') >= first_day - 365 - 28)].copy()
    train_bool = (train_copy.index.get_level_values('d') < first_day) \
                 & (train_copy.index.get_level_values('d') >= first_day - 365) & train_copy[target_col].notnull() & (train_copy['month'] != 12)
    if use_test:
        test_bool = (train_copy.index.get_level_values('d') >= first_day) \
                    & (train_copy.index.get_level_values('d') <= last_predict) & train_copy[target_col].notnull()
    blank_days = 0
    for d in range(first_day, last_day + 1):
        print(d)
        prediction_ind = train_copy.index.get_level_values('d') == d
        train_copy.loc[prediction_ind, 'prediction'] = est.predict(train_copy.loc[prediction_ind, train_cols])
        blank_days += 1
        _, train_copy = calc_lags(train_copy, blank_days)
        lgb_train = lgb.Dataset(train_copy.loc[train_bool, train_cols], label=train_copy.loc[train_bool, 'value_copy'])
        if use_test:
            lgb_test = lgb.Dataset(train_copy.loc[test_bool, train_cols], label=train_copy.loc[test_bool, 'value_copy'])
            valid_sets = [lgb_train, lgb_test]
            valid_names = ['train', 'valid']
        else:
            valid_sets = [lgb_train]
            valid_names = ['train']
        est = lgb.train(params, lgb_train, valid_sets=valid_sets, valid_names=valid_names, num_boost_round=num_rounds,
                        categorical_feature=cat_cols)
    return train_copy.loc[(train_copy.index.get_level_values('d') >= first_day) & (train_copy.index.get_level_values('d') <= last_day), 'prediction']


# train on last year
last_predict = 1941
first_predict = last_predict - 27
target_col = 'value'
train_bool = (train.index.get_level_values('d') < first_predict) & (train.index.get_level_values('d') >= first_predict - 365) & train[target_col].notnull() & (train['month'] != 12)
lgb_train = lgb.Dataset(train.loc[train_bool, train_cols], label=train.loc[train_bool, target_col])
test_bool = (train.index.get_level_values('d') >= first_predict) & (train.index.get_level_values('d') <= last_predict) & train[target_col].notnull()
lgb_test = lgb.Dataset(train.loc[test_bool, train_cols], label=train.loc[test_bool, target_col])
valid_sets = [lgb_train, lgb_test]
valid_names = ['train', 'valid']
params = {'objective': 'regression', 'learning_rate': 0.05, 'num_leaves': 64, 'bagging_fraction': 0.8, 'lambda_l2': 1,
              'bagging_freq': 1, 'seed': 0, 'verbose': -1}
est = lgb.train(params, lgb_train, valid_sets=valid_sets, valid_names=valid_names, num_boost_round=5000,
                early_stopping_rounds=20, categorical_feature=cat_cols)
best_iter = est.best_iteration
feat_imp_arr = est.feature_importance()
feat_imp = {f: feat_imp_arr[i] for i, f in enumerate(train_cols)}
# prediction
prediction_valid = predict(first_predict, last_predict, train, est, best_iter, True)
valid_ind = (train.index.get_level_values('d') >= first_predict) & (train.index.get_level_values('d') <= last_predict) & train[target_col].notnull()
prediction_valid.loc[prediction_valid < 0] = 0
prediction_valid = prediction_valid.to_frame('value')
print(mean_squared_error(train.loc[valid_ind, 'value_copy'], prediction_valid.loc[valid_ind, 'value']))
if not test_mode:
    # retrain using last month of data
    last_predict = 1969
    first_predict = 1942
    _, train = calc_aggs(train)
    train_bool = (train.index.get_level_values('d') < first_predict) & (train.index.get_level_values('d') >= first_predict - 365) & train[target_col].notnull() & (train['month'] != 12)
    lgb_train = lgb.Dataset(train.loc[train_bool, train_cols], label=train.loc[train_bool, target_col])
    est = lgb.train(params, lgb_train, num_boost_round=best_iter, categorical_feature=cat_cols)
    prediction_eval = predict(first_predict, last_predict, train, est, best_iter, False)
    prediction_eval.loc[prediction_eval < 0] = 0
    prediction_eval = prediction_eval.to_frame('value')
    # pivot the predictions
    prediction_valid.reset_index(inplace=True)
    prediction_valid['id'] = enc_dict['item_id'].inverse_transform(prediction_valid['item_id']) \
                             + '_' + enc_dict['store_id'].inverse_transform(prediction_valid['store_id']) \
                             + '_validation'
    prediction_valid.drop(['item_id', 'store_id', 'dept_id'], axis=1, inplace=True)
    prediction_valid_pivot = prediction_valid.pivot(columns='d', values='value', index='id')
    prediction_valid_pivot.columns = ['F' + str(i) for i in range(1, 29)]
    prediction_eval.reset_index(inplace=True)
    prediction_eval['id'] = enc_dict['item_id'].inverse_transform(prediction_eval['item_id']) \
                            + '_' + enc_dict['store_id'].inverse_transform(prediction_eval['store_id']) \
                            + '_evaluation'
    prediction_eval.drop(['item_id', 'store_id', 'dept_id'], axis=1, inplace=True)
    prediction_eval_pivot = prediction_eval.pivot(columns='d', values='value', index='id')
    prediction_eval_pivot.columns = ['F' + str(i) for i in range(1, 29)]
    # combine and output
    output = pd.concat([prediction_eval_pivot, prediction_valid_pivot], axis=0)
    output.reset_index(inplace=True)
    output.to_csv('submission_{}.csv'.format(dt.datetime.now().strftime('%y%m%d_%H%M')), index=False)


