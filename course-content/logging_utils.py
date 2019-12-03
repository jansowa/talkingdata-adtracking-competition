from datetime import datetime


def create_logging_path():
    current_date = datetime.now()
    return "model-results/model-from-" + current_date.strftime("%Y-%m-%d-%H.%M.%S") + ".txt"


def log(text, path):
    print(text, file=open(path, "a"))


def print_log(text, path):
    print(text)
    print(text, file=open(path, "a"))


def log_model_results(feature_columns, param, num_round, verbose_eval, early_stopping_rounds, valid_score, test_score=None):
    file_path = create_logging_path()
    log("Feature columns: ", file_path)
    log(str(feature_columns), file_path)
    log("LightGBM params:", file_path)
    log(param, file_path)
    log("num_round: " + str(num_round), file_path)
    log("verbose_eval: " + str(verbose_eval), file_path)
    log("early_stopping_rounds: " + str(early_stopping_rounds), file_path)
    print_log("Validation AUC score: " + str(valid_score), file_path)
    if test_score is not None:
        print_log("Test score: " + str(test_score), file_path)
