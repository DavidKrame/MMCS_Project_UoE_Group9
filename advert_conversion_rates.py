import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from functools import cache


def create_genre_vector(movie_genres: set[str], all_genres: list[str]) -> list[int]:
    '''
    "One hot encodes" the genres of a movie into the vector space of all
    possible genres.
    '''
    
    return [1 if genre in movie_genres else 0 for genre in all_genres]

@cache
def genre_overlap_score(genre_set_A: set[str], genre_set_B: set[str], all_genres: list[str]) -> float:
    '''
    Works out how similar two movies A & B are, based on the
    cosine similarity of their genre vectors.
    '''
    genre_vec_A = create_genre_vector(movie_genres=genre_set_A, all_genres=all_genres)
    genre_vec_B = create_genre_vector(movie_genres=genre_set_B, all_genres=all_genres)

    return cosine_similarity([genre_vec_A], [genre_vec_B])[0, 0]


def calculate_conversion_rate(schedule_df: pd.DataFrame, movie_df: pd.DataFrame,
                              all_movie_genres: list[str], ad_time_slot: str,
                              movie_title: str, max_conversion_rate: float) -> float:
    '''
    This works out how many people that are shown an advert for a movie
    will be likely to watch that movie.  This will depend on the current
    audience's tastes, i.e. is there an overlap between the genres of the
    movie currently being shown and the movie being advertised.

    Takes into account scenarios where an advert straddles the end of one
    movie and the start of another.  Here audience interests is assumed to
    be an average of the two movies.

    This returns a stochastic value between 5-100% of the maximum conversion
    rate, drawn from a normal distribution centred on the overlap score.
    Maximum conversion rate is currently.

    NOTE: THIS ASSUMES AD BREAKS ARE 1 BLOCK SIZE IN DURATION.
    NOTE: Think this function should be used by the students in their codebase somehow.

    :param schedule_df: Time-indexed pandas dataframe containing the tv schedule.
    :param movie_df: Pandas dataframe containing the movie data.
    :param all_movie_genres: List of strings of all unique movie genres in database.   
    :param ad_time_slot: string representing timestamp of selected advert spot
                       : in YYYY-MM-DD HH:mm:SS format.
    :param movie_title: Title of the advertised movie.
    :param max_conversion_rate: Maximum expected fraction of audience that would
                              : watch a movie advertised to them.
    '''

    assert ad_time_slot in schedule_df.index, "Selected time slot not found in index."
    assert schedule_df.loc[ad_time_slot].content_type == 'Advert', "Selected time slot is not an advert."
    assert movie_df.title.eq(movie_title).any(), "Selected movie not found in database."

    ad_slot_index = schedule_df.index.get_loc(ad_time_slot)

    # find movies either side of the ad slot, handle edge cases
    previous_content = schedule_df.iloc[ad_slot_index - 1].content if ad_slot_index > 0 else None
    next_content = schedule_df.iloc[ad_slot_index + 1].content if ad_slot_index < (len(schedule_df) - 1) else None

    previous_genres = set(movie_df.query('title == @previous_content').genres.iloc[0]) if previous_content else set()
    next_genres = set(movie_df.query('title == @next_content').genres.iloc[0]) if next_content else set()
    advertised_genres = set(movie_df.query('title == @movie_title').genres.iloc[0])

    overlap_scores = []

    if len(previous_genres) > 0:
        overlap_scores.append(genre_overlap_score(tuple(previous_genres),
                                                  tuple(advertised_genres),
                                                  all_movie_genres))
    else:
        pass

    if len(next_genres) > 0:
        overlap_scores.append(genre_overlap_score(tuple(next_genres),
                                                  tuple(advertised_genres),
                                                  all_movie_genres))
    else:
        pass

    overlap_score = np.mean(overlap_scores)
    stochastic_conversion_rate = np.clip(np.random.normal(loc=overlap_score, scale=0.1),
                                         a_min=0.05, a_max=1.)
    
    return stochastic_conversion_rate * max_conversion_rate


def calculate_genre_conversion_rate(schedule_df: pd.DataFrame, movie_df: pd.DataFrame,
                                    all_movie_genres: list[str], ad_time_slot: str,
                                    advert_genre: str, max_conversion_rate: float) -> float:
    '''
    Similar to calculate_conversion_rate, but do individual genres for adverts 
    rather than a s

    :param schedule_df: Time-indexed pandas dataframe containing the tv schedule.
    :param movie_df: Pandas dataframe containing the movie data.
    :param all_movie_genres: List of strings of all unique movie genres in database.   
    :param ad_time_slot: string representing timestamp of selected advert spot
                       : in YYYY-MM-DD HH:mm:SS format.
    :param advert_genre: Genre of the advertised movie.
    :param max_conversion_rate: Maximum expected fraction of audience that would
                              : watch a movie advertised to them.
    '''

    assert ad_time_slot in schedule_df.index, "Selected time slot not found in index."
    assert schedule_df.loc[ad_time_slot].content_type == 'Advert', "Selected time slot is not an advert."
    # assert tmdb_data.title.eq(movie_title).any(), "Selected movie not found in database."

    ad_slot_index = schedule_df.index.get_loc(ad_time_slot)

    # find movies either side of the ad slot, handle edge cases
    previous_content = schedule_df.iloc[ad_slot_index - 1].content if ad_slot_index > 0 else None
    next_content = schedule_df.iloc[ad_slot_index + 1].content if ad_slot_index < (len(schedule_df) - 1) else None

    previous_genres = set(movie_df.query('title == @previous_content').genres.iloc[0]) if previous_content else set()
    next_genres = set(movie_df.query('title == @next_content').genres.iloc[0]) if next_content else set()
    advertised_genres = [advert_genre] # set(movie_df.query('title == @movie_title').genres.iloc[0])

    overlap_scores = []

    if len(previous_genres) > 0:
        overlap_scores.append(genre_overlap_score(tuple(previous_genres),
                                                  tuple(advertised_genres),
                                                  all_movie_genres))
    else:
        pass

    if len(next_genres) > 0:
        overlap_scores.append(genre_overlap_score(tuple(next_genres),
                                                  tuple(advertised_genres),
                                                  all_movie_genres))
    else:
        pass

    overlap_score = np.mean(overlap_scores)
    
    return overlap_score * max_conversion_rate
