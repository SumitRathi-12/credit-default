import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, f1_score, roc_auc_score

def cross_validate(model, X, y, folds, seed):
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    m = {"recall": [], "f1": [], "roc_auc": []}

    for tr, te in skf.split(X, y):
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[te])[:, 1]
        yhat = (p >= 0.5).astype(int)

        m["recall"].append(recall_score(y.iloc[te], yhat))
        m["f1"].append(f1_score(y.iloc[te], yhat))
        m["roc_auc"].append(roc_auc_score(y.iloc[te], p))

    return {k: (np.mean(v), np.std(v)) for k, v in m.items()}
