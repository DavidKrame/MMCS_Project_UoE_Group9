import pandas as pd

def get_sample_of_movie_db(file_path, output_file, sample_size):
    """
    Load dataset as df and get reduced random sample of size sample_size

    write this to a new csv
    """

    df = pd.read_csv(file_path, parse_dates=['release_date'])
    reduced_df = df.sample(n=sample_size, random_state=1)

    reduced_df.to_csv(output_file, index=False)

sample_size = 1000
get_sample_of_movie_db('data/movie_database.csv', f'data/reduced_movie_database_{sample_size}.csv', sample_size)