import os
from pathlib import Path 
from collections import defaultdict

import pandas as pd

def preprocess_movielens(max_len=50):
    ratings = pd.read_csv(
    os.path.join(Path().resolve(), "datasets/ml-1m/ratings.dat"),
    sep="::",
    engine="python",
    header=None,
    names=["user_id", "movie_id", "rating", "timestamp"]
)
    ratings = ratings.sort_values(by=["user_id", "timestamp"])
    user_seq = defaultdict(list)

    for _, row in ratings.iterrows():
        user_seq[row["user_id"]].append(row["movie_id"])

    user_seq = list(user_seq.values())

    return user_seq

