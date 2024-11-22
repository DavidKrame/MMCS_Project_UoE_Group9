import warnings
import pandas as pd
warnings.filterwarnings('ignore')

three_day_no_budget_df = pd.read_csv('output/output_3LatDays_OurMovies_2024-11-21_20-37-20.738063.csv')
movie_db_df = pd.read_csv('data/filtered_movie_database_weekend.csv', parse_dates=['release_date'])

movies_shown = three_day_no_budget_df.loc[
    three_day_no_budget_df['Action'] == "Show Movie",
    ["Movie Title"]
]

movies_shown_ls = movies_shown["Movie Title"].unique()

movies_shown_db_df = movie_db_df[movie_db_df['title'].isin(movies_shown_ls)]

license_fee_cost = movies_shown_db_df['license_fee'].sum()

print(license_fee_cost)

# print(three_day_no_budget_df.at[0,'Viewerships'])

# #print(total_cost)

# sold_ad_df = three_day_no_budget_df.loc[
#     (three_day_no_budget_df['Channel'] == 'Own Channel') &
#     (three_day_no_budget_df['Action'] == 'Sell Adslot')
# ]

# sold_ad_df['Viewerships'] = (sold_ad_df['Viewerships'] / 1000).astype(int) *1000

# print(sold_ad_df['Viewerships'].sum())
