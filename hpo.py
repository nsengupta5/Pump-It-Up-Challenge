import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from initializer import get_column_transformer
from sklearn.preprocessing import FunctionTransformer

SEED = 42
TRIALS = 100
x_train = None
x_test = None
y_train = None

def rf_objective(trial):
    global x_train, x_test
    num_encoder = trial.suggest_categorical('num_encoder', ['StandardScaler', 'Manual', 'None'])
    cat_encoder = trial.suggest_categorical('cat_encoder', ['OneHotEncoder', 'OrdinalEncoder', 'TargetEncoder', 'Manual'])
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
    )

    train_pipeline = make_pipeline(column_transformer, clf)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(train_pipeline, x_train, y_train, cv=skf, scoring="accuracy")

    return scores.mean()

def lr_objective(trial):
    num_encoder = trial.suggest_categorical('num_encoder', ['StandardScaler', 'Manual', 'None'])
    cat_encoder = trial.suggest_categorical('cat_encoder', ['OneHotEncoder', 'OrdinalEncoder', 'TargetEncoder', 'Manual'])
    C = trial.suggest_float('C', 0.1, 10)
    class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
    solver = trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga', 'newton-cholesky'])
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

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(train_pipeline, x_train, y_train, cv=skf, scoring="accuracy")

    return scores.mean()

def gb_objective(trial):
    num_encoder = trial.suggest_categorical('num_encoder', ['StandardScaler', 'Manual', 'None'])
    cat_encoder = trial.suggest_categorical('cat_encoder', ['OneHotEncoder', 'OrdinalEncoder', 'TargetEncoder', 'Manual'])
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 100)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 100)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    subsample = trial.suggest_float('subsample', 0.5, 1.0)

    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        subsample=subsample,
        learning_rate=learning_rate,
        random_state=SEED,
    )

    column_transformer = get_column_transformer(x_train, cat_encoder, num_encoder)
    train_pipeline = make_pipeline(column_transformer, clf)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(train_pipeline, x_train, y_train, cv=skf, scoring="accuracy")

    return scores.mean()

def hist_gb_objective(trial):
    num_encoder = trial.suggest_categorical('num_encoder', ['StandardScaler', 'Manual', 'None'])
    cat_encoder = trial.suggest_categorical('cat_encoder', ['OneHotEncoder', 'OrdinalEncoder', 'TargetEncoder', 'Manual'])
    max_iter = trial.suggest_int('max_iter', 100, 1000)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2)
    max_depth = trial.suggest_int('max_depth', 3, 30)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 20, 100)
    l2_regularization = trial.suggest_float('l2_regularization', 0.0, 1.0)
    max_bins = trial.suggest_int('max_bins', 10, 255)

    clf = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        max_bins=max_bins,
        random_state=SEED,
    )

    handle_sparse_transformer = FunctionTransformer(
        lambda x: x.toarray() if hasattr(x, "toarray") else x,
        accept_sparse=True,
    )
    column_transformer = get_column_transformer(x_train, cat_encoder, num_encoder)
    train_pipeline = make_pipeline(column_transformer, handle_sparse_transformer, clf)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(train_pipeline, x_train, y_train, cv=skf, scoring="accuracy")

    return scores.mean()

def mlp_objective(trial):
    num_encoder = trial.suggest_categorical('num_encoder', ['StandardScaler', 'Manual', 'None'])
    cat_encoder = trial.suggest_categorical('cat_encoder', ['OneHotEncoder', 'OrdinalEncoder', 'TargetEncoder', 'Manual'])
    hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [(100,), (100, 100), (100, 100, 100)])
    activation = trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu'])
    solver = trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam'])
    alpha = trial.suggest_float('alpha', 0.0001, 0.1)
    learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive'])
    max_iter = trial.suggest_int('max_iter', 100, 1000)

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=SEED,
    )

    column_transformer = get_column_transformer(x_train, cat_encoder, num_encoder)
    train_pipeline = make_pipeline(column_transformer, clf)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(train_pipeline, x_train, y_train, cv=skf, scoring="accuracy")

    return scores.mean()


def get_best_hyperparams(x, y, model):
    print("----------------- FINDING BEST HYPERPARAMETERS -----------------")
    global x_train, x_test, y_train, column_transformer
    x_train = x
    y_train = y

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{model}_tuning",
        storage=f"sqlite:///{model}_tuning.db",
        load_if_exists=True,
    )

    if model == "RandomForestClassifier":
        study.optimize(rf_objective, n_trials=TRIALS)
    elif model == "LogisticRegression":
        study.optimize(lr_objective, n_trials=TRIALS)
    elif model == "GradientBoostingClassifier":
        study.optimize(gb_objective, n_trials=TRIALS)
    elif model == "HistGradientBoostingClassifier":
        study.optimize(hist_gb_objective, n_trials=TRIALS)
    elif model == "MLPClassifier":
        study.optimize(mlp_objective, n_trials=TRIALS)
    else:
        raise ValueError(f"Invalid model type: {model}")

    print(f"Best hyperparameters: {study.best_params}")

    # Move the database to the correct location
    print("----------------- BEST HYPERPARAMETERS FOUND -----------------\n")
    return study.best_params
