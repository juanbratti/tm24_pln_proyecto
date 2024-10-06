import pandas as pd
from src.utils import parse_reviews_txt_to_csv


# preprocess raw data into a pandas dataframe
file_path = 'Arts.txt'

# get filepath to parsed file
processed_file_path = parse_reviews_txt_to_csv(file_path)
