import pandas as pd
import numpy as np
import xpress as xp

def movie_views_for_time_slot(x, i, j, movie_db_df, my_channel_df, Demos, population):
    """"
    Calculate the baseline viewership for movie i at time slot j
    """
    views = x[i][j] * xp.Sum(
                    (population*movie_db_df[f'{k}_scaled_popularity'].iloc[i])
                    *my_channel_df[f'{k}_baseline_view_count'].iloc[j]
                    for k in Demos)
    return views

def comp_advertised_views_for_time_slot(z, i, j, movie_db_df, channel_df, Demos, population, genres_conversion_df):
    """
    Calculate the viewership gained from adveretising movie i on channel c at time slot j 
    """
    views = z[i][j]*xp.Sum(
        (population * movie_db_df[f'{k}_scaled_popularity'].iloc[i])
        *channel_df[f'{k}_baseline_view_count'].loc[j]
        *genres_conversion_df[str(j)].loc[i]
        for k in Demos
        )
    return views

def own_advertised_views_for_time_slot(w, i , j, movie_db_df, my_channel_df, Demos, population):
    """
    Calculate the viewership gained from advertising movie i on own channel at time slot j
    conversion rate value is a place holder, fourther research pending
    """
    conversions = xp.Sum(movie_db_df[f'{k}_scaled_popularity'].iloc[i]*my_channel_df[f'{k}_baseline_view_count'].iloc[j] for k in Demos)
    total_viewers = xp.Sum(my_channel_df[f'{k}_baseline_view_count'].iloc[j] for k in Demos)

    views = w[i][j] * (conversions/total_viewers)* xp.Sum(
                    (population*movie_db_df[f'{k}_scaled_popularity'].iloc[i])
                    *my_channel_df[f'{k}_baseline_view_count'].iloc[j]
                    for k in Demos)
    
    return views

def calculate_ad_slot_price(j, schedule_df):
    base_fee = 10000
    profit_margin = 0.2
    budget_factor = 0.002
    box_office_factor = 0.001

    license_fee = (base_fee
                   + (budget_factor * schedule_df['movie_budget'].loc[j])
                   + (box_office_factor * schedule_df['box_office_revenue'].loc[j])
                   ) * (1. + profit_margin)

    ad_slot_cost = (license_fee / schedule_df['n_ad_breaks'].loc[j]) * schedule_df['prime_time_factor'].loc[j]

    return np.round(ad_slot_cost, 2)