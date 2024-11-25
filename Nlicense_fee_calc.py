import numpy as np
import pandas as pd

def calculate_ad_slot_price(schedule_df: pd.DataFrame, base_fee: float,
                            profit_margin: float, budget_factor: float,
                            box_office_factor: float) -> pd.Series:
    '''
    Works out the cost required to buy a specific ad slot.  This is based on the time
    of day, and the budget/earnings of the movie being shown before the
    chosen ad slot.

    This function is applied to a schedule dataframe to create a new column
    containing the ad slot prices, returns NaN if the slot is not an ad slot.

    This is also multiplied by the prime time factor, desired profit margin does
    not take into account the effects of prime time factor currently, i.e.
    there'll be a larger profit margin obtained than the one specified for spots
    in prime time.

    Values used in generation of dataset.
    base_fee = 10_000
    profit_margin = 0.2
    budget_factor = 0.002
    box_office_revenue_factor = 0.001

    :param schedule_df: Dataframe containing the populated schedule with movies and
                      : ad breaks.
    :param base_fee: Base fee required for all movies to be licensed to a channel
    :param profit_margin: Percent (in 0-1 scale) of license fee that the channel
                        : wants to make in profit.
    :param budget_factor: What percent (in 0-1 scale) of the movie's budget contributes
                        : to the license fee.
    :param box_office_factor: What percent (in 0-1 scale) of the movie's box office renvenue
                            : contributes to the license fee.
    '''

    license_fee = (base_fee
                   + (budget_factor * schedule_df.movie_budget)
                   + (box_office_factor * schedule_df.box_office_revenue)
                   ) * (1. + profit_margin)

    ad_slot_cost = (license_fee / schedule_df.n_ad_breaks) * schedule_df.prime_time_factor

    return np.round(ad_slot_cost, 2)