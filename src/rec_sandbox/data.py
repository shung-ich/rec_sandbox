import os
from pathlib import Path 
import pandas as pd

ratings_df = pd.read_csv(
    os.path.join(Path().resolve(), "datasets/ml-1m/ratings.dat"),
    sep="::",
    engine="python",
    header=None,
    names=["uu_id", "movie_id", "rating", "timestamp"]
)
print(ratings_df.head())
# import os
# def get_relative_path():
#     current_dir = os.path.dirname(__file__)
#     datasets_path = os.path.join(current_dir, '..', '..', 'datasets')
#     return os.path.abspath(datasets_path)
# pd.read_csv()
