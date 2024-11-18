import numpy as np
import pandas as pd
from datetime import datetime
def create_conversion_rate_csv(output_path, conversion_rates_df, movie_genre_df, number_of_movies, Movies, Ad_slots, number_of_ad_slots):

    conversion_rates_movie = np.array(
        [sum(conversion_rates_df[g].loc[j]*movie_genre_df[g].loc[i] for g in Genres) for i in Movies for j in Ad_slots]
    ).reshape(number_of_movies, number_of_ad_slots)

    conversion_rate_df = pd.DataFrame(conversion_rates_movie)

    conversion_rate_df.to_csv(output_path)

if __name__ == "__main__":
    channel_0_df = pd.read_csv('data/ADVERTS_FIRST_WEEK_channel_0_schedule.csv', parse_dates=['Date-Time'])
    channel_1_df = pd.read_csv('data/ADVERTS_FIRST_WEEK_channel_1_schedule.csv', parse_dates=['Date-Time'])
    channel_2_df = pd.read_csv('data/ADVERTS_FIRST_WEEK_channel_2_schedule.csv', parse_dates=['Date-Time'])

    conversion_rates_0_df = pd.read_csv('data/FIRST_WEEK_channel_0_conversion_rates.csv', parse_dates=['Date-Time'])
    conversion_rates_1_df = pd.read_csv('data/FIRST_WEEK_channel_1_conversion_rates.csv', parse_dates=['Date-Time'])
    conversion_rates_2_df = pd.read_csv('data/FIRST_WEEK_channel_2_conversion_rates.csv', parse_dates=['Date-Time'])

    early_cutoff = datetime(2024, 10, 4, 0, 0, 0)
    late_cutoff = datetime(2024, 10, 5, 0, 0, 0)

    conversion_rates_0_df = conversion_rates_0_df.drop(conversion_rates_0_df[(conversion_rates_0_df['Date-Time'] > late_cutoff) & (conversion_rates_0_df['Date-Time'] < early_cutoff)].index)
    conversion_rates_1_df = conversion_rates_1_df.drop(conversion_rates_1_df[(conversion_rates_1_df['Date-Time'] > late_cutoff) & (conversion_rates_1_df['Date-Time'] < early_cutoff)].index)
    conversion_rates_2_df = conversion_rates_2_df.drop(conversion_rates_2_df[(conversion_rates_2_df['Date-Time'] > late_cutoff) & (conversion_rates_2_df['Date-Time'] < early_cutoff)].index)

    movie_db_df = pd.read_csv('data/movie_database_with_license_fee_100.csv', parse_dates=['release_date'])
    movie_genre_df = pd.read_csv('data/movie_genre_hot_one_100.csv')

    # movie_db_df = pd.read_csv('data/movie_database_with_license_fee_300.csv', parse_dates=['release_date'])
    # movie_genre_df = pd.read_csv('data/movie_genre_hot_one_300.csv')

    number_of_movies = len(movie_db_df.index)

    number_of_ad_slots_0 = len(conversion_rates_0_df)
    number_of_ad_slots_1 = len(conversion_rates_1_df)
    number_of_ad_slots_2 = len(conversion_rates_2_df)

    Movies = range(number_of_movies)
    Genres = movie_genre_df.columns.to_list()

    Ad_slots_0 = range(number_of_ad_slots_0)
    Ad_slots_1 = range(number_of_ad_slots_1)
    Ad_slots_2 = range(number_of_ad_slots_2)

    
    # output_0 = 'data/movies_adslots_conversion_0.csv'
    # output_1 = 'data/movies_adslots_conversion_1.csv'
    # output_2 = 'data/movies_adslots_conversion_2.csv'

    output_0 = 'data/movies_adslots_conversion_0_100.csv'
    output_1 = 'data/movies_adslots_conversion_1_100.csv'
    output_2 = 'data/movies_adslots_conversion_2_100.csv'

    create_conversion_rate_csv(output_0, conversion_rates_0_df, movie_genre_df, number_of_movies, Movies, Ad_slots_0, number_of_ad_slots_0)
    create_conversion_rate_csv(output_1, conversion_rates_1_df, movie_genre_df, number_of_movies, Movies, Ad_slots_1, number_of_ad_slots_1)
    create_conversion_rate_csv(output_2, conversion_rates_2_df, movie_genre_df, number_of_movies, Movies, Ad_slots_2, number_of_ad_slots_2)
