from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

def build_pipeline(prep, model, smote, seed):
    steps = [("prep", prep)]
    if smote:
        steps.append(("smote", SMOTE(random_state=seed)))
        return ImbPipeline(steps + [("model", model)])
    return Pipeline(steps + [("model", model)])
