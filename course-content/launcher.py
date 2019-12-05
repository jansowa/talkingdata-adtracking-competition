import pandas as pd
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from itertools import combinations
from processing_utils import get_data_splits
from processing_utils import train_model

train_sample_path = '../competition-data/train_sample.csv'
train_sample_data = pd.read_csv(train_sample_path, parse_dates=['click_time'])

# print(train_sample_data.head())

# Add columns with day, hour, minute, second
clicks = train_sample_data.copy()
clicks['day'] = clicks['click_time'].dt.day.astype('uint8')
clicks['hour'] = clicks['click_time'].dt.hour.astype('uint8')
clicks['minute'] = clicks['click_time'].dt.minute.astype('uint8')
clicks['second'] = clicks['click_time'].dt.second.astype('uint8')

categorical_features = ['ip', 'app', 'device', 'os', 'channel']

# Add columns with label encoding
label_encoder = LabelEncoder()
for feature in categorical_features:
    clicks[feature + "_labels"] = label_encoder.fit_transform(clicks[feature])

interactions = pd.DataFrame(index=clicks.index)
for first_cat, second_cat in combinations(categorical_features, 2):
    new_column = clicks[first_cat].astype(str) + "_" + clicks[second_cat].astype(str)
    new_column = new_column.rename(first_cat + "_" + second_cat)
    interactions = interactions.join(new_column)

interactions_columns = []
for column in interactions.columns:
    interactions[column] = label_encoder.fit_transform(interactions[column])
    interactions_columns.append(column)

clicks = clicks.join(interactions)

train, valid, test = get_data_splits(clicks, valid_fraction=0.1)

for each in [train, valid, test]:
    print(f"Outcome fraction = {each.is_attributed.mean():.6f}")

# Add columns with count encoding
count_encoder = ce.CountEncoder(cols=categorical_features)
count_encoder.fit(train[categorical_features])

train = train.join(count_encoder.transform(train[categorical_features]).add_suffix('_count'))
valid = valid.join(count_encoder.transform(valid[categorical_features]).add_suffix('_count'))
test = test.join(count_encoder.transform(test[categorical_features]).add_suffix('_count'))

# Add columns with target encoding
target_encoder = ce.TargetEncoder(cols=categorical_features)
target_encoder.fit(train[categorical_features], train['is_attributed'])

train = train.join(target_encoder.transform(train[categorical_features]).add_suffix('_target'))
valid = valid.join(target_encoder.transform(valid[categorical_features]).add_suffix('_target'))
test = test.join(target_encoder.transform(test[categorical_features]).add_suffix('_target'))

# Add columns with CatBoost encoding
catboost_encoder = ce.CatBoostEncoder(cols=categorical_features)
catboost_encoder.fit(train[categorical_features], train['is_attributed'])

train = train.join(catboost_encoder.transform(train[categorical_features]).add_suffix('_catboost'))
valid = valid.join(catboost_encoder.transform(valid[categorical_features]).add_suffix('_catboost'))
test = test.join(catboost_encoder.transform(test[categorical_features]).add_suffix('_catboost'))

# Add column with clicks in last 6 hours
train = train.sort_values('click_time')
valid = valid.sort_values('click_time')
test = test.sort_values('click_time')

train_time_sorted = pd.Series(train.index, index=train.click_time, name="count_6_hours").sort_index()
valid_time_sorted = pd.Series(valid.index, index=valid.click_time, name="count_6_hours").sort_index()
test_time_sorted = pd.Series(test.index, index=test.click_time, name="count_6_hours").sort_index()

train_count_6_hours = train_time_sorted.rolling('6h').count() - 1
valid_count_6_hours = valid_time_sorted.rolling('6h').count() - 1
test_count_6_hours = test_time_sorted.rolling('6h').count() - 1

train_count_6_hours.index = train.index
valid_count_6_hours.index = valid.index
test_count_6_hours.index = test.index

train['ip_past_6hr_counts'] = train_count_6_hours
valid['ip_past_6hr_counts'] = valid_count_6_hours
test['ip_past_6hr_counts'] = test_count_6_hours

# Add columns with time since last click of corresponding ip
train['timedeltas'] = train.groupby('ip')['click_time'].diff()
valid['timedeltas'] = valid.groupby('ip')['click_time'].diff()
test['timedeltas'] = test.groupby('ip')['click_time'].diff()

train['timedeltas'] = train['timedeltas'].apply(lambda time: time.total_seconds())
valid['timedeltas'] = valid['timedeltas'].apply(lambda time: time.total_seconds())
test['timedeltas'] = test['timedeltas'].apply(lambda time: time.total_seconds())

# print("BASELINE MODEL:")
# feature_cols = ['day', 'hour', 'minute', 'second', 'ip', 'app', 'device', 'os',
#                 'channel']
# train_model(train, valid, test, feature_cols, early_stopping_rounds=30)
#
# print("LABEL ENCODING:")
# feature_cols = ['day', 'hour', 'minute', 'second', 'ip_labels', 'app_labels', 'device_labels', 'os_labels',
#                 'channel_labels']
# train_model(train, valid, test, feature_cols, early_stopping_rounds=30)

print("COUNT ENCODING:")
feature_cols = ['day', 'hour', 'minute', 'second', 'ip_count', 'app_count', 'device_count', 'os_count',
                'channel_count', 'ip_past_6hr_counts', 'timedeltas']
final_model, _, _ = train_model(train, valid, test, feature_cols, early_stopping_rounds=30)

# print("TARGET ENCODING:")
# feature_cols = ['day', 'hour', 'minute', 'second', 'ip_target', 'app_target', 'device_target', 'os_target',
#                 'channel_target']
# train_model(train, valid, test, feature_cols, early_stopping_rounds=30)
#
# print("CATBOOST ENCODING:")
# feature_cols = ['day', 'hour', 'minute', 'second', 'ip_catboost', 'app_catboost', 'device_catboost', 'os_catboost',
#                 'channel_catboost', 'timedeltas']
# train_model(train, valid, test, feature_cols, early_stopping_rounds=30)

# Build solution
test_sample_path = '../competition-data/test.csv'
test_X = pd.read_csv(train_sample_path, parse_dates=['click_time'])

test_X['day'] = test_X['click_time'].dt.day.astype('uint8')
test_X['hour'] = test_X['click_time'].dt.hour.astype('uint8')
test_X['minute'] = test_X['click_time'].dt.minute.astype('uint8')
test_X['second'] = test_X['click_time'].dt.second.astype('uint8')

categorical_features = ['ip', 'app', 'device', 'os', 'channel']

# Add columns with label encoding
label_encoder = LabelEncoder()
for feature in categorical_features:
    test_X[feature + "_labels"] = label_encoder.fit_transform(test_X[feature])

interactions = pd.DataFrame(index=test_X.index)
for first_cat, second_cat in combinations(categorical_features, 2):
    new_column = test_X[first_cat].astype(str) + "_" + test_X[second_cat].astype(str)
    new_column = new_column.rename(first_cat + "_" + second_cat)
    interactions = interactions.join(new_column)

interactions_columns = []
for column in interactions.columns:
    interactions[column] = label_encoder.fit_transform(interactions[column])
    interactions_columns.append(column)

test_X = test_X.join(interactions)

# Add columns with count encoding
count_encoder = ce.CountEncoder(cols=categorical_features)
count_encoder.fit(test_X[categorical_features])

test_X = test_X.join(count_encoder.transform(test_X[categorical_features]).add_suffix('_count'))

# Add column with clicks in last 6 hours
test_X = test_X.sort_values('click_time')
test_X_time_sorted = pd.Series(test_X.index, index=test_X.click_time, name="count_6_hours").sort_index()
test_X_count_6_hours = test_X_time_sorted.rolling('6h').count() - 1
test_X_count_6_hours.index = test_X.index
test_X['ip_past_6hr_counts'] = test_X_count_6_hours

# Add columns with time since last click of corresponding ip
test_X['timedeltas'] = test_X.groupby('ip')['click_time'].diff()
test_X['timedeltas'] = test_X['timedeltas'].apply(lambda time: time.total_seconds())

# Predict data
feature_cols = ['day', 'hour', 'minute', 'second', 'ip_count', 'app_count', 'device_count', 'os_count',
                'channel_count', 'ip_past_6hr_counts', 'timedeltas']
test_X.sort_index(inplace=True)
test_predictions = final_model.predict(test_X[feature_cols])

# Format output data to proper style
output = pd.DataFrame({'click_id': test_X.index,
                       'is_attributed': test_predictions})
output.to_csv('./model-results/my_submission.csv', index=False)
