from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


def optimize_models(
    X,
    y,
    preprocessor,
    models,
    param_grids,
    scoring
):

    results = []

    for name, model in models.items():

        if name not in param_grids:
            continue

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grids[name],
            n_iter=3,
            cv=3,
            scoring=scoring,
            random_state=42,
            n_jobs=-1,
            refit=True
        )

        search.fit(X, y)

        results.append({
            "Model": name,
            "Best Score": search.best_score_,
            "Best Parameters": search.best_params_,
            "Best Estimator": search.best_estimator_
        })

    return results