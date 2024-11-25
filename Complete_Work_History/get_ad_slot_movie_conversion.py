import numpy as np
import pandas as pd
from datetime import datetime
def create_conversion_rate_csv(output_path, conversion_rates_df, movie_genre_df, number_of_movies, Movies, Ad_slots, number_of_ad_slots, Genres):

    conversion_rates_movie = np.array(
        [sum(conversion_rates_df[g].iloc[j]*movie_genre_df[g].loc[i] for g in Genres) for i in Movies for j in Ad_slots]
    ).reshape(number_of_movies, number_of_ad_slots)

    conversion_rate_df = pd.DataFrame(conversion_rates_movie)

    conversion_rate_df.to_csv(output_path)

if __name__ == "__main__":

    conversion_rates_0_df = pd.read_csv('data/FIRST_WEEK_channel_0_conversion_rates.csv', parse_dates=['Date-Time'])
    conversion_rates_1_df = pd.read_csv('data/FIRST_WEEK_channel_1_conversion_rates.csv', parse_dates=['Date-Time'])
    conversion_rates_2_df = pd.read_csv('data/FIRST_WEEK_channel_2_conversion_rates.csv', parse_dates=['Date-Time'])

    cutoff = datetime(2024, 10, 5, 0, 0, 0)

    conversion_4block_0_df = conversion_rates_0_df.drop(conversion_rates_0_df[conversion_rates_0_df['Date-Time'] > cutoff].index)
    conversion_4block_1_df = conversion_rates_1_df.drop(conversion_rates_1_df[conversion_rates_1_df['Date-Time'] > cutoff].index)
    conversion_4block_2_df = conversion_rates_2_df.drop(conversion_rates_2_df[conversion_rates_2_df['Date-Time'] > cutoff].index)

    conversion_3block_0_df = conversion_rates_0_df.drop(conversion_rates_0_df[conversion_rates_0_df['Date-Time'] < cutoff].index)
    conversion_3block_1_df = conversion_rates_1_df.drop(conversion_rates_1_df[conversion_rates_1_df['Date-Time'] < cutoff].index)
    conversion_3block_2_df = conversion_rates_2_df.drop(conversion_rates_2_df[conversion_rates_2_df['Date-Time'] < cutoff].index)

    movie_db_4block_df = pd.read_csv('data/filtered_movie_database_working_days.csv', parse_dates=['release_date'])
    movie_genre_4block_df = pd.read_csv('data/filtered_movie_database_genre_hot_one_working_days.csv')

    movie_db_3block_df = pd.read_csv('data/filtered_movie_database_weekend.csv', parse_dates=['release_date'])
    movie_genre_3block_df = pd.read_csv('data/filtered_movie_database_genre_hot_one_weekend.csv')

    # movie_db_df = pd.read_csv('data/movie_database_with_license_fee_300.csv', parse_dates=['release_date'])
    # movie_genre_df = pd.read_csv('data/movie_genre_hot_one_300.csv')

    number_of_movies_4block = len(movie_db_4block_df.index)
    number_of_movies_3block = len(movie_db_3block_df.index)

    number_of_ad_slots_0_4block = len(conversion_4block_0_df)
    number_of_ad_slots_1_4block = len(conversion_4block_1_df)
    number_of_ad_slots_2_4block = len(conversion_4block_2_df)

    number_of_ad_slots_0_3block = len(conversion_3block_0_df)
    number_of_ad_slots_1_3block = len(conversion_3block_1_df)
    number_of_ad_slots_2_3block = len(conversion_3block_2_df)

    Movies_4block = range(number_of_movies_4block)
    Movies_3block = range(number_of_movies_3block)
    Genres_4block = movie_genre_4block_df.columns.to_list()
    Genres_3block = movie_genre_3block_df.columns.to_list()

    Ad_slots_0_4block = range(number_of_ad_slots_0_4block)
    Ad_slots_1_4block = range(number_of_ad_slots_1_4block)
    Ad_slots_2_4block = range(number_of_ad_slots_2_4block)

    Ad_slots_0_3block = range(number_of_ad_slots_0_3block)
    Ad_slots_1_3block = range(number_of_ad_slots_1_3block)
    Ad_slots_2_3block = range(number_of_ad_slots_2_3block)

    
    # output_0 = 'data/movies_adslots_conversion_0.csv'
    # output_1 = 'data/movies_adslots_conversion_1.csv'
    # output_2 = 'data/movies_adslots_conversion_2.csv'

    output_0_4block = 'data/movie_adslots_conversion_0_4block.csv'
    output_1_4block = 'data/movie_adslots_conversion_1_4block.csv'
    output_2_4block = 'data/movie_adslots_conversion_2_4block.csv'

    output_0_3block = 'data/movie_adslots_conversion_0_3block.csv'
    output_1_3block = 'data/movie_adslots_conversion_1_3block.csv'
    output_2_3block = 'data/movie_adslots_conversion_2_3block.csv'

    create_conversion_rate_csv(output_0_4block, conversion_4block_0_df, movie_genre_4block_df, number_of_movies_4block, Movies_4block, Ad_slots_0_4block, number_of_ad_slots_0_4block, Genres_4block)
    create_conversion_rate_csv(output_1_4block, conversion_4block_1_df, movie_genre_4block_df, number_of_movies_4block, Movies_4block, Ad_slots_1_4block, number_of_ad_slots_1_4block, Genres_4block)
    create_conversion_rate_csv(output_2_4block, conversion_4block_2_df, movie_genre_4block_df, number_of_movies_4block, Movies_4block, Ad_slots_2_4block, number_of_ad_slots_2_4block, Genres_4block)

    create_conversion_rate_csv(output_0_3block, conversion_3block_0_df, movie_genre_3block_df, number_of_movies_3block, Movies_3block, Ad_slots_0_3block, number_of_ad_slots_0_3block, Genres_3block)
    create_conversion_rate_csv(output_1_3block, conversion_3block_1_df, movie_genre_3block_df, number_of_movies_3block, Movies_3block, Ad_slots_1_3block, number_of_ad_slots_1_3block, Genres_3block)
    create_conversion_rate_csv(output_2_3block, conversion_3block_2_df, movie_genre_3block_df, number_of_movies_3block, Movies_3block, Ad_slots_2_3block, number_of_ad_slots_2_3block, Genres_3block)
