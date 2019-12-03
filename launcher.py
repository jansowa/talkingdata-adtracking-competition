import pandas as pd
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

train_sample_path = './competition-data/train_sample.csv'
train_sample_data = pd.read_csv(train_sample_path, parse_dates=['click_time'])

print(train_sample_data.head())

clicks = train_sample_data.copy()
clicks['day'] = clicks['click_time'].dt.day.astype('uint8')
clicks['hour'] = clicks['click_time'].dt.hour.astype('uint8')
clicks['minute'] = clicks['click_time'].dt.minute.astype('uint8')
clicks['second'] = clicks['click_time'].dt.second.astype('uint8')

categorical_features = ['ip', 'app', 'device', 'os', 'channel']
encoder = LabelEncoder()

for feature in categorical_features:
    clicks[feature+"_labels"] = encoder.fit_transform(clicks[feature])

feature_cols = ['day', 'hour', 'minute', 'second', 'ip_labels', 'app_labels', 'device_labels', 'os_labels', 'channel_labels']

valid_fraction = 0.1
clicks_sorted = clicks.sort_values('click_time')
valid_rows = int(len(clicks_sorted) * valid_fraction)

train = clicks_sorted[:-valid_rows*2]
test = clicks_sorted[-valid_rows*2:-valid_rows]
valid = clicks_sorted[-valid_rows:]

for each in [train, valid, test]:
    print(f"first Outcome fraction = {each.is_attributed.mean():.6f}")

# better division - choose after some experiments
# train = clicks_sorted[valid_rows:-valid_rows]
# test = clicks_sorted[:valid_rows]
# valid = clicks_sorted[-valid_rows:]
#
# for each in [train, valid, test]:
#     print(f"third Outcome fraction = {each.is_attributed.mean():.6f}")

dtrain = lgb.Dataset(train[feature_cols], label=train['is_attributed'])
dvalid = lgb.Dataset(valid[feature_cols], label=valid['is_attributed'])
dtest = lgb.Dataset(test[feature_cols], label=test['is_attributed'])

param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 1000
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=30)