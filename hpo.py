import optuna

def objective(trial):
    evaluation_score = 0.0

    return evaluation_score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

importances = optuna.importance.get_param_importances(study)
print(importances)
