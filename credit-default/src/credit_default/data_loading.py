from pathlib import Path
import pandas as pd
from typing import Tuple

def load_german(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, sep=" ", header=None)
    X = df.iloc[:, :-1]
    y = (df.iloc[:, -1] == 2).astype(int)
    return X, y

def load_credit_card(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    target = next(c for c in df.columns if "default" in c.lower())
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    return X, y
