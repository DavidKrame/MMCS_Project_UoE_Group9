import pandas as pd
import xpress as xp
import ast
import datetime as dt
from time import time

xp.init('C:/xpressmp/bin/xpauth.xpr')
start_time = time()
# dataset loading
# my_channel_df = pd.read_csv('data/FIRST_WEEK_channel_A_schedule.csv', parse_dates=['Date-Time'])
# movie_db_df = pd.read_csv('data/movie_database_with_license_fee.csv', parse_dates=['release_date'])
# other_channels_0_df = pd.read_csv('data/FIRST_WEEK_channel_0_schedule.csv', parse_dates=['Date-Time'])
# other_channels_1_df = pd.read_csv('data/FIRST_WEEK_channel_1_schedule.csv', parse_dates=['Date-Time'])
# other_channels_2_df = pd.read_csv('data/FIRST_WEEK_channel_2_schedule.csv', parse_dates=['Date-Time'])
# conversion_rates_0_df = pd.read_csv('data/FIRST_WEEK_channel_0_conversion_rates.csv', parse_dates=['Date-Time'])
# conversion_rates_1_df = pd.read_csv('data/FIRST_WEEK_channel_1_conversion_rates.csv', parse_dates=['Date-Time'])
# conversion_rates_2_df = pd.read_csv('data/FIRST_WEEK_channel_2_conversion_rates.csv', parse_dates=['Date-Time'])

my_channel_df = pd.read_csv('data/AGGREGATE_FIRST_WEEK_channel_A_schedule.csv', parse_dates=['Date-Time'])
movie_db_df = pd.read_csv('data/movie_database_with_license_fee_1000.csv', parse_dates=['release_date'])
other_channels_0_df = pd.read_csv('data/AGGREGATE_FIRST_WEEK_channel_0_schedule.csv', parse_dates=['Date-Time'])
other_channels_1_df = pd.read_csv('data/AGGREGATE_FIRST_WEEK_channel_1_schedule.csv', parse_dates=['Date-Time'])
other_channels_2_df = pd.read_csv('data/AGGREGATE_FIRST_WEEK_channel_2_schedule.csv', parse_dates=['Date-Time'])
conversion_rates_0_df = pd.read_csv('data/AGGREGATE_FIRST_WEEK_channel_0_conversion_rates.csv', parse_dates=['Date-Time'])
conversion_rates_1_df = pd.read_csv('data/AGGREGATE_FIRST_WEEK_channel_1_conversion_rates.csv', parse_dates=['Date-Time'])
conversion_rates_2_df = pd.read_csv('data/AGGREGATE_FIRST_WEEK_channel_2_conversion_rates.csv', parse_dates=['Date-Time'])

# model initialization
model = xp.problem()
print('problem intialised at time ', time() - start_time)
# Decision Variables for Movie Scheduling
movie_indices = movie_db_df.index.tolist()
x = [[xp.var(name=f"x_{i}_{j}", vartype=xp.binary) for j in my_channel_df['Date-Time']] for i in movie_indices]
model.addVariable([var for sublist in x for var in sublist])  # Flatten and add all variables to the model

# Decision Variables for Advertising
ad_indices = ['Channel_0', 'Channel_1', 'Channel_2']
ad_vars = {ch: xp.var(name=f"ad_{ch}", vartype=xp.binary) for ch in ad_indices}
model.addVariable(list(ad_vars.values()))
# model.addVariable(ad_vars.values())  # Add advertising variables to the model
print('first set of desicion vars added at time ', time() - start_time)

# auxiliary binary variables to track genre presence per day
my_channel_df['Day'] = my_channel_df['Date-Time'].dt.date
genre_presence = {
    (day, genre): xp.var(vartype=xp.binary, name=f"genre_presence_{day}_{genre}")
    for day in my_channel_df['Day'].unique()
    for genre in movie_db_df['genres'].explode().unique()
}
print('desicion vars genre presence per day added at time ', time() - start_time)

# Add the auxiliary variables to the model
model.addVariable(list(genre_presence.values()))

# Objective Function: Maximization of the total viewership's revenue minus costs
viewership_from_movies = xp.Sum(x[i][j] * movie_db_df['scaled_popularity'].iloc[i] for i in movie_indices for j in range(len(my_channel_df)))
print('viewership_from_movies intialised at time ', time() - start_time)
# Create a mapping of conversion rates for each channel
conversion_rates_mapping = {
    'Channel_0': conversion_rates_0_df,
    'Channel_1': conversion_rates_1_df,
    'Channel_2': conversion_rates_2_df
}
movie_db_df['genres'] = movie_db_df['genres'].apply(ast.literal_eval)
# viewership_from_ads = xp.Sum(
#     ad_vars[ch] * xp.Sum(
#         x[i][j] * conversion_rates_mapping[ch].loc[j, genre]  # Usage of genre-based conversion rates
#         for j in range(len(my_channel_df))
#         for genre in list(movie_db_df.loc[i, 'genres'])  # Iterating through genres for each movie
#     )
#     for i in movie_indices for ch in ad_indices
# )
# print('viewrship_from_ads intialised at time ', time() - start_time)
# Total costs
license_fees = xp.Sum(x[i][j] * movie_db_df['license_fee'].iloc[i] for i in movie_indices for j in range(len(my_channel_df)))
print('license_fee intialised at time ', time() - start_time)
ad_costs = xp.Sum(ad_vars[ch] * (other_channels_0_df['ad_slot_price'].sum() if ch == 'Channel_0' else
                                  other_channels_1_df['ad_slot_price'].sum() if ch == 'Channel_1' else
                                  other_channels_2_df['ad_slot_price'].sum())
                  for ch in ad_indices)  # Add advertising costs
print('ad_cost intialised at time ', time() - start_time)

# Objective Function: Maximize total viewership minus costs
# model.setObjective(viewership_from_movies + viewership_from_ads - license_fees - ad_costs, sense=xp.maximize)
model.setObjective(viewership_from_movies - license_fees - ad_costs, sense=xp.maximize)
print('objective function set at time ', time() - start_time)

# Constraints

No_of_Time_slots = range(len(my_channel_df))

# 1. Time slot constraint: You can only schedule one movie per time slot
time_slots = my_channel_df['Date-Time'].unique()
model.addConstraint(xp.Sum(x[i][j] for i in movie_indices) == 1 for j in No_of_Time_slots)
print('constraint 1 added at time ', time() - start_time)

# 2. Movie must be scheduled for its whole time
# Ensure the movie is scheduled for its entire runtime if it is scheduled at time slot j
# STILL NEED TO LOOK AT THIS CAREFULLY
model.addConstraint(
    xp.Sum(x[i][j + k] for k in range(movie_db_df['runtime'].iloc[i] // 30) if j + k < len(my_channel_df)) == x[i][j] * movie_db_df['runtime'].iloc[i]
    for i in movie_indices for j in No_of_Time_slots
    )
print('constraint 2 added at time ', time() - start_time)

# 3. Total runtime constraint: Total scheduled runtime should not exceed a limit (24 hours for us)
max_runtime = 24 * 60  # in minutes
model.addConstraint(xp.Sum(x[i][j] * movie_db_df['runtime'].iloc[i] for i in movie_indices for j in No_of_Time_slots) <= max_runtime)
print('constraint 3 added at time ', time() - start_time)

# 4. Consecutive time slots constraint
for i in movie_indices:
    for j in range(len(my_channel_df)):
        for k in range(j + 1, len(my_channel_df)):
            final_time = my_channel_df['Date-Time'].iloc[k]
            initial_time = my_channel_df['Date-Time'].iloc[j]
            slot_duration = (final_time - initial_time).total_seconds() / 60
            model.addConstraint(x[i][j] * slot_duration <= x[i][j] * movie_db_df['runtime'].iloc[i])
print('constraint 4 added at time ', time() - start_time)

# 5. Budget constraint for movies
total_budget = 1000000  # Example budget
model.addConstraint(
    xp.Sum(x[i][j] * movie_db_df['budget'].iloc[i] for i in movie_indices for j in range(len(my_channel_df))) <= total_budget
)
print('constraint 5 added at time ', time() - start_time)

# 6. Advertising budget constraint
total_ad_budget = 50000  # Example advertising budget
model.addConstraint(
    xp.Sum(ad_vars[ch] * (other_channels_0_df['ad_slot_price'].sum() if ch == 'Channel_0' else
                          other_channels_1_df['ad_slot_price'].sum() if ch == 'Channel_1' else
                          other_channels_2_df['ad_slot_price'].sum())
               for ch in ad_indices) <= total_ad_budget
)
print('constraint 6 added at time ', time() - start_time)

# 7. Threshold for Conversion Rates

# need new decision variable for movie i advertised pon channel c at slot t

conversion_rate_threshold = 0.2  # Example threshold
for i in movie_indices:
    for j in No_of_Time_slots:
        for ch in ad_indices:
            # Single check: Ensure j is within bounds for conversion_rates_mapping[ch] and genre exists
            genre = movie_db_df['genres'].iloc[i][0]  # Assume the first genre is primary

            if j < len(conversion_rates_mapping[ch]) and genre in conversion_rates_mapping[ch].columns:
                model.addConstraint(
                    ad_vars[ch] * x[i][j] * conversion_rates_mapping[ch].iloc[j][genre] >=
                    ad_vars[ch] * x[i][j] * conversion_rate_threshold
                )

print('constraint 7 added at time ', time() - start_time)

# 8. Daily Genre Diversity Constraint
max_genres_per_day = 3   # maximum number of genres allowed per day

# genre presence constraints and diversity constraint
for day in my_channel_df['Day'].unique():
    # Get all time slots for the current day
    daily_slots = my_channel_df[my_channel_df['Day'] == day].index.tolist()
    
    for genre in movie_db_df['genres'].explode().unique():
        # Enforce that genre_presence[day, genre] is set to 1 if the genre is scheduled at least once on this day
        model.addConstraint(
            xp.Sum(
                x[i][j] for i in movie_indices for j in daily_slots if genre in movie_db_df['genres'].iloc[i]
            ) >= genre_presence[day, genre]
        )
    
    # Total genres per day should not exceed max_genres_per_day
    model.addConstraint(
        xp.Sum(genre_presence[day, genre] for genre in movie_db_df['genres'].explode().unique()) <= max_genres_per_day
    )

print('Constraint 8: Daily genre diversity constraint added at time :', time() - start_time)

# # 9. Genre Clashes Constraint
# competitor_schedules = pd.concat([
#     other_channels_0_df[['Date-Time', 'content_type']],
#     other_channels_1_df[['Date-Time', 'content_type']],
#     other_channels_2_df[['Date-Time', 'content_type']]
# ], ignore_index=True)

# # mapping from movie titles to genres in the movie database
# title_to_genre = {row['title']: row['genres'] for index, row in movie_db_df.iterrows()}

# # genre info added to the competitor schedules based on titles
# competitor_schedules['genre'] = competitor_schedules['content_type'].map(title_to_genre)

# # keep only the scheduled movies (not advertisements)
# competitor_movies = competitor_schedules[competitor_schedules['content_type'] == 'Movie']

# # constraints to avoid genre clashes
# for i in movie_indices:
#     movie_title = movie_db_df['title'].iloc[i]  # title of the movie to be scheduled
#     movie_genres = set(title_to_genre[movie_title])  # retrieve genres of the movie and convert to a set

#     for j in range(len(my_channel_df)):
#         if my_channel_df['Date-Time'].iloc[j] is not None:
#             # Find all competitor movies scheduled at the same time
#             competing_movies = competitor_movies[competitor_movies['Date-Time'] == my_channel_df['Date-Time'].iloc[j]]
            
#             # now, constraints to limit genre clashes
#             for _, competing_movie in competing_movies.iterrows():
#                 competing_movie_genres = set(competing_movie['genre'])  # Get genres of the competing movie
                
#                 # Check for genre clash using intersection
#                 if movie_genres.intersection(competing_movie_genres):
#                     model.addConstraint(x[i][j] + xp.Sum(x[other_i][j] for other_i in movie_indices 
#                                                           if set(title_to_genre[movie_db_df['title'].iloc[other_i]]).intersection(competing_movie_genres)) > 0 <= 1)

# print('Constraint 9: Genre clashes constraint added at time ', time() - start_time)
print('Constraint 9: Genre clashes TO DO AFTER INTEGRATING 1ST WEEK DATASET')

# Solve the model
model.solve()
print('model solved at time ', time() - start_time)

# Output the results for scheduled movies
for i in movie_indices:
    for j in range(len(my_channel_df)):
        if x[i][j].getSolution() > 0.5:  # Movie is scheduled
            scheduled_time = my_channel_df['Date-Time'].iloc[j]
            print(f"Scheduled Movie: {movie_db_df['title'].iloc[i]}, Time Slot: {scheduled_time}")

# Output the results for advertising
for ch in ad_indices:
    if ad_vars[ch].getSolution() > 0.5:  # Advertising on this channel
        print(f"Advertising on {ch}")

# Optionally, display the objective value
print("Maximized Viewership:", model.getObjVal())
