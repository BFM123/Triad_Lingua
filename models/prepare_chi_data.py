import argparse
import os
import shutil
import subprocess
import tempfile

import pandas as pd
from datasets import Dataset, DatasetDict


def find_parallel_csv(root_dir):
    # look for obvious CSV filenames
    candidates = ["parallel_corpus.csv", "parallel_corpus.txt", "corpus.csv"]
    for cand in candidates:
        path = os.path.join(root_dir, cand)
        if os.path.exists(path):
            return path
    # fallback: first csv file in tree
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".csv"):
                return os.path.join(dirpath, fn)
    return None


def load_csv_as_df(path: str) -> pd.DataFrame:
    # try several encodings and separators
    for sep in [",", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8")
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue
    # last resort: pandas autodetect
    return pd.read_csv(path, encoding="utf-8", engine="python")


def choose_columns(df: pd.DataFrame):
    cols = list(df.columns)
    # try to find chichewa and english columns by name
    src_candidates = [c for c in cols if any(x in c.lower() for x in ["chichew", "chewa", "ny", "nyanja", "chi"]) ]
    tgt_candidates = [c for c in cols if any(x in c.lower() for x in ["english", "en"]) ]
    if src_candidates and tgt_candidates:
        return src_candidates[0], tgt_candidates[0]
    # otherwise take first two columns
    return cols[0], cols[1]


def prepare_dataset(repo_url: str, out_dir: str, split_ratio: float = 0.95):
    tmp = tempfile.mkdtemp(prefix="chi_corpus_")
    try:
        print("Cloning repository...")
        subprocess.check_call(["git", "clone", "--depth", "1", repo_url, tmp])

        csv_path = find_parallel_csv(tmp)
        if not csv_path:
            raise FileNotFoundError("No csv parallel file found in the repository. Please inspect the repo manually.")

        print(f"Using parallel file: {csv_path}")
        df = load_csv_as_df(csv_path)
        src_col, tgt_col = choose_columns(df)
        df = df[[src_col, tgt_col]].dropna()
        df.columns = ["source", "target"]

        # basic cleanup: strip whitespace
        df["source"] = df["source"].astype(str).str.strip()
        df["target"] = df["target"].astype(str).str.strip()

        # dedupe
        df = df.drop_duplicates()

        ds = Dataset.from_pandas(df)
        ds = ds.shuffle(seed=42)

        n = len(ds)
        train_n = int(n * split_ratio)
        train_ds = ds.select(range(0, train_n))
        val_ds = ds.select(range(train_n, n))

        dataset_dict = DatasetDict({"train": train_ds, "validation": val_ds})
        os.makedirs(out_dir, exist_ok=True)
        dataset_dict.save_to_disk(out_dir)
        print(f"Saved dataset to {out_dir} (train={len(train_ds)}, val={len(val_ds)})")

    finally:
        shutil.rmtree(tmp)


def main():
    parser = argparse.ArgumentParser(description="Download and prepare Chichewa-English parallel data")
    parser.add_argument("--repo-url", type=str, default="https://github.com/avtaylor/SpokenChichewaCorpus.git")
    parser.add_argument("--out-dir", type=str, default="models/data/chichewa_en")
    parser.add_argument("--split-ratio", type=float, default=0.95)
    args = parser.parse_args()

    prepare_dataset(args.repo_url, args.out_dir, args.split_ratio)


if __name__ == "__main__":
    main()
