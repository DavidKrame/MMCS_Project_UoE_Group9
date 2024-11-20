import pandas as pd

def get_sample_of_movie_db(file_path, reduced_df_output_file, genre_hot_one_output_file, sample_size):
    """
    Load dataset as df and get reduced random sample of size sample_size

    write this to a new csv
    """

    df = pd.read_csv(file_path, parse_dates=['release_date'])
    reduced_df = df.sample(n=sample_size, random_state=1)

    reduced_df.to_csv(reduced_df_output_file, index=False)

    reduced_df['genres'] = reduced_df['genres'].apply(eval)
    
    # Use pd.get_dummies to create the one-hot encoded dataframe
    genres_df = reduced_df['genres'].str.join('|').str.get_dummies()

    genres_df.to_csv(genre_hot_one_output_file, index=False)
    



sample_size = 135
# get_sample_of_movie_db('data/movie_database.csv', f'data/reduced_movie_database_{sample_size}.csv', sample_size)
# get_sample_of_movie_db(
#     'data/movie_database_with_license_fee.csv',
#     f'data/movie_database_with_license_fee_{sample_size}.csv',
#     f'data/movie_genre_hot_one_{sample_size}.csv',
#     sample_size
#     )

get_sample_of_movie_db(
    'data/filtered_movie_database.csv',
    f'data/filtered_movie_database_{sample_size}.csv',
    f'data/filtered_movie_database_genre_hot_one_{sample_size}.csv',
    sample_size
    )