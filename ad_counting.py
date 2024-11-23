import warnings
import pandas as pd

import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

three_day_no_budget_df = pd.read_csv('output/output_3LatDays_OurMovies_2024-11-21_20-37-20.738063.csv')

channel_0_df = pd.read_csv('data/ADVERTS_FIRST_WEEK_channel_0_schedule.csv', parse_dates=['Date-Time'])
channel_1_df = pd.read_csv('data/ADVERTS_FIRST_WEEK_channel_1_schedule.csv', parse_dates=['Date-Time'])
channel_2_df = pd.read_csv('data/ADVERTS_FIRST_WEEK_channel_2_schedule.csv', parse_dates=['Date-Time'])

no_ads_on_own = three_day_no_budget_df.loc[
    (three_day_no_budget_df['Action'] == "Advertise Movie") &
    (three_day_no_budget_df['Channel'] == "Own Channel"),
    ['Action']
].count()

no_ads_sold = three_day_no_budget_df.loc[
    (three_day_no_budget_df['Action'] == "Sell Adslot") &
    (three_day_no_budget_df['Channel'] == "Own Channel"),
    ['Action']
].count()

bought_comp_slots = no_ads_sold = three_day_no_budget_df.loc[
    (three_day_no_budget_df['Action'] == "Advertise Movie") &
    (three_day_no_budget_df['Channel'] != "Own Channel"),
    ['Action']
].count()

total_comp_slots = len(channel_0_df.index) + len(channel_1_df.index) + len(channel_2_df.index)

print(no_ads_on_own) #56
print(no_ads_sold) #46
print(bought_comp_slots) #20
print(total_comp_slots) #667