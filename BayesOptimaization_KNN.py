from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def KNN_cv(n_neighbors, x, y):
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    cval = cross_val_score(estimator, x, y, scoring='accuracy', cv=5)
    return cval.mean()


def optimize_KNN(x, y):
    def KNN_crossval(n_neighbors):
        return KNN_cv(n_neighbors, x=x, y=y)

    optimizer = BayesianOptimization(
        f=KNN_crossval,
        pbounds={"n_neighbors": (0, 10)},
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(n_iter=10)

    print("Final result:", optimizer.max)
    return optimizer.max