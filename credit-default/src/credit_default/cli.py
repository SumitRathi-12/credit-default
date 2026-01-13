import argparse
from pathlib import Path
from .experiments import run

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["german", "credit_card"])
    p.add_argument("--smote", choices=["on", "off"], default="off")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="outputs")
    a = p.parse_args()

    run(a.dataset, a.smote == "on", a.cv_folds, a.seed, Path(a.outdir) / a.dataset)

if __name__ == "__main__":
    main()
