import warnings
import pandas as pd

import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

files = [
    'output/output_3LatDays_OurMovies_2024-11-21_20-37-20.738063.csv',
    'output/Sentivity_output_3LastDays_2024-11-23_08-58-35.053220_budget_30100000.csv',
    'output/Sentivity_output_3LastDays_2024-11-23_11-59-47.870448_budget_30050000.csv',
    'output/Sentivity_output_3LastDays_2024-11-23_17-03-54.846653_budget_30000000.csv',
    'output/Sentivity_output_3LastDays_2024-11-23_23-41-48.705903_budget_29100000.csv',
    'output/Sentivity_output_3LastDays_2024-11-24_03-41-41.287624_budget_29050000.csv',
    # 'output/Sentivity_output_3LastDays_2024-11-23_23-41-48.705903_budget_29000000.csv',
    # 'output/Sentivity_output_3LastDays_2024-11-23_23-41-48.705903_budget_28050000.csv'
]

# three_day_no_budget_df = pd.read_csv('output/output_3LatDays_OurMovies_2024-11-21_20-37-20.738063.csv')

channel_0_df = pd.read_csv('data/ADVERTS_FIRST_WEEK_channel_0_schedule.csv', parse_dates=['Date-Time'])
channel_1_df = pd.read_csv('data/ADVERTS_FIRST_WEEK_channel_1_schedule.csv', parse_dates=['Date-Time'])
channel_2_df = pd.read_csv('data/ADVERTS_FIRST_WEEK_channel_2_schedule.csv', parse_dates=['Date-Time'])


for file in files:
    schedule_df = pd.read_csv(file)

    no_ads_on_own = schedule_df.loc[
        (schedule_df['Action'] == "Advertise Movie") &
        (schedule_df['Channel'] == "Own Channel"),
        ['Action']
    ].count()

    no_ads_sold = schedule_df.loc[
        (schedule_df['Action'] == "Sell Adslot") &
        (schedule_df['Channel'] == "Own Channel"),
        ['Action']
    ].count()

    bought_comp_slots = no_ads_sold = schedule_df.loc[
        (schedule_df['Action'] == "Advertise Movie") &
        (schedule_df['Channel'] != "Own Channel"),
        ['Action']
    ].count()

    total_comp_slots = len(channel_0_df.index) + len(channel_1_df.index) + len(channel_2_df.index)

    # print(no_ads_on_own) #56
    # print(no_ads_sold) #46
    # print(bought_comp_slots) #20
    # print(total_comp_slots) #667
    print(f"Results for file: {file}")
    print("Number of ads on own channel:", no_ads_on_own) 
    print("Number of ads sold on own channel:", no_ads_sold)
    print("Bought ad slots on competitor channels:", bought_comp_slots)
    print("Total ad slots for competitor channels:", total_comp_slots)
    print("-" * 50) # separtor