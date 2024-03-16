import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from initializer import get_column_transformer

SEED = 42
TRIALS = 100
x_train = None
y_train = None

"""
Objective function for the RandomForestClassifier

args:
    trial: The trial to optimize the hyperparameters

returns:
    The mean accuracy of the model
"""
def rf_objective(trial):
    global x_train, y_train
    num_encoder = trial.suggest_categorical('num_encoder', 
                                            ['StandardScaler', 'Manual', 'None'])
    cat_encoder = trial.suggest_categorical('cat_encoder', 
                                            ['OneHotEncoder', 'OrdinalEncoder', 'TargetEncoder', 'Manual'])
    n_estimators = trial.suggest_int('n_estimators', 50, 400)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    column_transformer = get_column_transformer(x_train, cat_encoder, num_encoder)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        max_features=max_features,
        random_state=SEED,
        n_jobs = 5
    )

    train_pipeline = make_pipeline(column_transformer, clf)

    # Use 5-fold cross-validation to get the mean accuracy
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(train_pipeline, x_train, y_train, cv=skf, scoring="accuracy")

    return scores.mean()

"""
Objective function for the LogisticRegression

args:
    trial: The trial to optimize the hyperparameters

returns:
    The mean accuracy of the model
"""
def lr_objective(trial):
    global x_train, y_train
    num_encoder = trial.suggest_categorical('num_encoder', 
                                            ['StandardScaler', 'Manual', 'None'])
    cat_encoder = trial.suggest_categorical('cat_encoder', 
                                            ['OneHotEncoder', 'OrdinalEncoder', 'TargetEncoder', 'Manual'])
    C = trial.suggest_float('C', 0.1, 10)
    class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
    solver = trial.suggest_categorical('solver', 
                                       ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga', 'newton-cholesky'])
    max_iter = trial.suggest_int('max_iter', 100, 2000)
    fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
    tol = trial.suggest_float('tol', 1e-5, 1e-1)

    clf = LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver=solver,
        max_iter=max_iter,
        fit_intercept=fit_intercept,
        tol=tol,
        random_state=SEED,
    )

    column_transformer = get_column_transformer(x_train, cat_encoder, num_encoder)

    train_pipeline = make_pipeline(column_transformer, clf)

    # Use 5-fold cross-validation to get the mean accuracy
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(train_pipeline, x_train, y_train, cv=skf, scoring="accuracy")

    return scores.mean()

def get_best_hyperparams(x, y, model):
    print("----------------- FINDING BEST HYPERPARAMETERS -----------------")
    global x_train, y_train, column_transformer
    x_train = x
    y_train = y

    # Create a study to store the best hyperparameters
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{model}_tuning",
        storage=f"sqlite:///{model}_tuning.db",
        load_if_exists=True,
    )

    # Commented out as best hyperparameters have been found
    # if model == "RandomForestClassifier":
    #     study.optimize(rf_objective, n_trials=TRIALS)
    # elif model == "LogisticRegression":
    #     study.optimize(lr_objective, n_trials=TRIALS)
    # elif model == "GradientBoostingClassifier":
    #     study.optimize(gb_objective, n_trials=TRIALS)
    # elif model == "HistGradientBoostingClassifier":
    #     study.optimize(hist_gb_objective, n_trials=TRIALS)
    # elif model == "MLPClassifier":
    #     study.optimize(mlp_objective, n_trials=TRIALS)
    # else:
    #     raise ValueError(f"Invalid model type: {model}")

    print(f"Best hyperparameters: {study.best_params}")
    print("----------------- BEST HYPERPARAMETERS FOUND -----------------\n")

    return study.best_params
