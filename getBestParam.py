from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def get_best_estimator(estimator, param, x, y):
    search = GridSearchCV(estimator=estimator, param_grid=param, cv=5)
    search.fit(x, y)
    print(f"最佳参数:{search.best_params_}")
    return search.best_params_
