import pandas as pd
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from processing_utils import get_data_splits
from processing_utils import train_model

train_sample_path = '../competition-data/train_sample.csv'
train_sample_data = pd.read_csv(train_sample_path, parse_dates=['click_time'])

print(train_sample_data.head())

clicks = train_sample_data.copy()
clicks['day'] = clicks['click_time'].dt.day.astype('uint8')
clicks['hour'] = clicks['click_time'].dt.hour.astype('uint8')
clicks['minute'] = clicks['click_time'].dt.minute.astype('uint8')
clicks['second'] = clicks['click_time'].dt.second.astype('uint8')

categorical_features = ['ip', 'app', 'device', 'os', 'channel']
label_encoder = LabelEncoder()

for feature in categorical_features:
    clicks[feature + "_labels"] = label_encoder.fit_transform(clicks[feature])

clicks_sorted = clicks.sort_values('click_time')

train, valid, test = get_data_splits(clicks_sorted, valid_fraction=0.2)

for each in [train, valid, test]:
    print(f"Outcome fraction = {each.is_attributed.mean():.6f}")

count_encoder = ce.CountEncoder(categorical_features)
count_encoder.fit(train[categorical_features])

train = train.join(count_encoder.transform(train[categorical_features]).add_suffix('_count'))
valid = valid.join(count_encoder.transform(valid[categorical_features]).add_suffix('_count'))
test = test.join(count_encoder.transform(test[categorical_features]).add_suffix('_count'))

print("LABEL ENCODING:")
feature_cols = ['day', 'hour', 'minute', 'second', 'ip_labels', 'app_labels', 'device_labels', 'os_labels',
                'channel_labels']

bst, valid_score, test_score = train_model(train, valid, test, feature_cols, early_stopping_rounds=30)

print("COUNT ENCODING:")
feature_cols = ['day', 'hour', 'minute', 'second', 'ip_count', 'app_count', 'device_count', 'os_count',
                'channel_count']

bst, valid_score, test_score = train_model(train, valid, test, feature_cols, early_stopping_rounds=30)