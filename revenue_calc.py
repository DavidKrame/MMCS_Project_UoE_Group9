import warnings
import pandas as pd

import matplotlib.pyplot as plt
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
total_cost = three_day_no_budget_df.at[0,'Viewerships']

# print(license_fee_cost)

print('Total cost', total_cost)

#print(total_cost)

sold_ad_df = three_day_no_budget_df.loc[
    (three_day_no_budget_df['Channel'] == 'Own Channel') &
    (three_day_no_budget_df['Action'] == 'Sell Adslot')
]
print('total viewes when ads sold', sold_ad_df['Viewerships'].sum())

sold_ad_df['Viewerships'] = (sold_ad_df['Viewerships'] / 1000).astype(int) *1000
rounded_views = sold_ad_df['Viewerships'].sum()

print('total viewes when ads sold rounded', rounded_views)

# license_fee = (10000
#                + (0.002 * movie_db_df['budget'].loc[1])
#                + (0.001 * movie_db_df['revenue'].loc[1])
#                ) * (1. + 0.2)

# print(license_fee)
# print(movie_db_df['license_fee'].loc[1])

selling_prices = [1.8, 1.82, 1.84, 1.86, 1.88, 1.9, 1.92, 1.94, 1.96, 1.98, 2]# [1.8, 2, 2.2, 2.4, 2.6, 2.8, 3] [1.8, 1.81, 1.82, 1.83, 1.84, 1.85, 1.86, 1.87, 1.88, 1.89, 1.9]

gp_dict = {}

print("\n")
for i in selling_prices:
    gross_profit_percent = 100*(rounded_views*i - total_cost)/(rounded_views*i)
    # print("Selling price: ", i)
    # print("Gross profit %: ", gross_profit_percent, "%")
    # print("\n")
    gp_dict[i] = round(gross_profit_percent, 3)

print(gp_dict)
# Extract keys and values
x = list(gp_dict.keys())
y = list(gp_dict.values())

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-', color='b', label='Data Points')

# Add titles and labels
plt.title('Plot of gp_dict', fontsize=16)
plt.xlabel('Selling ', fontsize=14)
plt.ylabel('Values', fontsize=14)

# Add grid
plt.grid(True)

# Add legend
plt.legend()

# Show the plot
# plt.show()

