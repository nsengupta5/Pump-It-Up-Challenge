import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

SEED = 42
x_train = None
y_train = None
column_transformer = None

def rf_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 400)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=SEED,
    )

    train_pipeline = make_pipeline(column_transformer, clf)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(train_pipeline, x_train, y_train, cv=skf, scoring="accuracy")

    return scores.mean()

def lr_objective(trial):
    evaluation_score = 0.0
    return evaluation_score

def gb_objective(trial):
    evaluation_score = 0.0
    return evaluation_score

def hist_gb_objective(trial):
    evaluation_score = 0.0
    return evaluation_score

def mlp_objective(trial):
    evaluation_score = 0.0
    return evaluation_score

def get_best_hyperparams(x, y, model, transformer):
    print("----------------- FINDING BEST HYPERPARAMETERS -----------------")
    global x_train, y_train, column_transformer
    x_train = x
    y_train = y
    column_transformer = transformer
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{model}_tuning",
        storage=f"sqlite:///{model}_tuning.db",
        load_if_exists=True,
    )
    if model == "RandomForestClassifier":
        study.optimize(rf_objective, n_trials=100)
    elif model == "LogisticRegression":
        study.optimize(lr_objective, n_trials=100)
    elif model == "GradientBoostingClassifier":
        study.optimize(gb_objective, n_trials=100)
    elif model == "HistGradientBoostingClassifier":
        study.optimize(hist_gb_objective, n_trials=100)
    elif model == "MLPClassifier":
        study.optimize(mlp_objective, n_trials=100)
    else:
        raise ValueError(f"Invalid model type: {model}")

    print(study.best_params)
    importances = optuna.importance.get_param_importances(study)
    print(importances)
    print("----------------- BEST HYPERPARAMETERS FOUND -----------------\n")
    return study.best_params
