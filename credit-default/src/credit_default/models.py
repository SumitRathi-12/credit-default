from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def get_models(seed):
    return {
        "logistic": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "svm": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)
    }
