from pathlib import Path
import pandas as pd
from .data_loading import load_german, load_credit_card
from .preprocessing import build_preprocessor
from .models import get_models
from .pipelines import build_pipeline
from .evaluation import cross_validate

def run(dataset, smote, folds, seed, outdir):
    if dataset == "german":
        X, y = load_german(Path("data/raw/german/german.data"))
    else:
        X, y = load_credit_card(Path("data/raw/credit_card/UCI_Credit_Card.csv"))

    prep = build_preprocessor(X)
    models = get_models(seed)
    rows = []

    for name, model in models.items():
        pipe = build_pipeline(prep, model, smote, seed)
        scores = cross_validate(pipe, X, y, folds, seed)
        rows.append({
            "model": name,
            "smote": smote,
            **{f"{k}_mean": v[0] for k, v in scores.items()},
            **{f"{k}_std": v[1] for k, v in scores.items()}
        })

    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "results.csv", index=False)
    df.to_markdown(outdir / "results.md", index=False)
