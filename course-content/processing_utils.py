import lightgbm as lgb
from sklearn import metrics
from logging_utils import log_model_results


def get_data_splits(dataframe, valid_fraction=0.1):
    # dataframe = dataframe.sort_values('click_time')
    valid_rows = int(len(dataframe) * valid_fraction)
    train = dataframe[:-valid_rows * 2].sort_values('click_time')
    valid = dataframe[-valid_rows * 2:-valid_rows].sort_values('click_time')
    test = dataframe[-valid_rows:].sort_values('click_time')

    return train, valid, test


def train_model(train, valid, test=None, feature_cols=None, early_stopping_rounds=20, verbose_eval=False,
                log_to_file=False):
    if feature_cols is None:
        feature_cols = train.columns.drop(['click_time', 'attributed_time', 'is_attributed'])

    dtrain = lgb.Dataset(train[feature_cols], label=train['is_attributed'])
    dvalid = lgb.Dataset(valid[feature_cols], label=valid['is_attributed'])

    param = {'num_leaves': 64, 'objective': 'binary', 'metric': 'auc', 'seed': 7, 'boost_from_average': False}
    num_round = 1000
    print("Training model!")
    bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid],
                    early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval)

    valid_pred = bst.predict(valid[feature_cols])
    valid_score = metrics.roc_auc_score(valid['is_attributed'], valid_pred)
    if not log_to_file:
        print(f"Validation AUC score: {valid_score}")

    if test is not None:
        test_pred = bst.predict(test[feature_cols])
        test_score = metrics.roc_auc_score(test['is_attributed'], test_pred)
        if log_to_file:
            log_model_results(feature_cols, param, num_round, verbose_eval, early_stopping_rounds, valid_score,
                              test_score)
        else:
            print(f"Test score: {test_score}")
        return bst, valid_score, test_score
    else:
        if log_to_file:
            log_model_results(feature_cols, param, num_round, verbose_eval, early_stopping_rounds, valid_score)
        return bst, valid_score
