from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# svc调参
def svc_cv(C, gamma, x, y):
    estimator = SVC(C=C, gamma=gamma, random_state=2)
    cval = cross_val_score(estimator, x, y, scoring='accuracy', cv=5)
    return cval.mean()


def optimize_svc(x, y):
    def svc_crossval(expC, expGamma):
        C = 10 ** expC
        gamma = 10 ** expGamma
        return svc_cv(C=C, gamma=gamma, x=x, y=y)

    optimizer = BayesianOptimization(
        f=svc_crossval,
        pbounds={"expC": (-3, 2), "expGamma": (-4, -1)},
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(n_iter=10)

    print("Final result:", optimizer.max)
    return optimizer.max

# KNN调参
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