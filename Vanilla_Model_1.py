import pandas as pd
import xpress as xp
from time import time

xp.init('C:/xpressmp/bin/xpauth.xpr')
start_time = time()
# dataset loading
my_channel_df = pd.read_csv('data/AGGREGATE_FIRST_WEEK_channel_A_schedule.csv', parse_dates=['Date-Time'])
movie_db_df = pd.read_csv('data/movie_database.csv', parse_dates=['release_date'])
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
print('desicion vars added at time ', time() - start_time)
# Objective Function: Maximization of the total viewership's revenue minus costs
viewership_from_movies = xp.Sum(x[i][j] * movie_db_df['scaled_popularity'].iloc[i] for i in movie_indices for j in range(len(my_channel_df)))
# Create a mapping of conversion rates for each channel
conversion_rates_mapping = {
    'Channel_0': conversion_rates_0_df,
    'Channel_1': conversion_rates_1_df,
    'Channel_2': conversion_rates_2_df
}
viewership_from_ads = xp.Sum(
    ad_vars[ch] * xp.Sum(
        x[i][j] * conversion_rates_mapping[ch].iloc[i][1:]  # Select conversion rates for the respective channel + Need to multiply by a certain cost
        for i in movie_indices for j in range(len(my_channel_df))
    )
    for ch in ad_indices
)
print('viewrship_from_ads intialised at time ', start_time - time())
# Total costs
license_fees = xp.Sum(x[i][j] * movie_db_df['license_fee'].iloc[i] for i in movie_indices for j in range(len(my_channel_df)))
print('license_fee intialised at time ', start_time - time())
ad_costs = xp.Sum(ad_vars[ch] * (other_channels_0_df['ad_slot_price'].sum() if ch == 'Channel_0' else
                                  other_channels_1_df['ad_slot_price'].sum() if ch == 'Channel_1' else
                                  other_channels_2_df['ad_slot_price'].sum())
                  for ch in ad_indices)  # Add advertising costs
print('ad_cost intialised at time ', start_time - time())

# Objective Function: Maximize total viewership minus costs
model.setObjective(viewership_from_movies + viewership_from_ads - license_fees - ad_costs, sense=xp.maximize)
print('objective function set at time ', time() - start_time)

# Constraints

# 1. Time slot constraint: You can only schedule one movie per time slot
time_slots = my_channel_df['Date-Time'].unique()
for j in range(len(my_channel_df)):
    model.addConstraint(xp.Sum(x[i][j] for i in movie_indices) <= 1, f"TimeConstraint_{j}")
print('constraint 1 added at time ', time() - start_time)

# 2. Movie must be scheduled for its whole time
for i in movie_indices:
    runtime = movie_db_df['runtime'].iloc[i]  # Movie runtime in minutes
    for j in range(len(my_channel_df)):
        # Ensure the movie is scheduled for its entire runtime if it is scheduled at time slot j
        model.addConstraint(xp.Sum(x[i][j + k] for k in range(runtime // 5) if j + k < len(my_channel_df)) == x[i][j] * runtime, f"FullRuntime_{i}_{j}")
print('constraint 2 added at time ', time() - start_time)

# 3. Total runtime constraint: Total scheduled runtime should not exceed a limit (24 hours for us)
max_runtime = 24 * 60  # in minutes
model.addConstraint(
    xp.Sum(x[i][j] * movie_db_df['runtime'].iloc[i] for i in movie_indices for j in range(len(my_channel_df))) <= max_runtime,
    "MaxRuntime"
)
print('constraint 3 added at time ', time() - start_time)

# 4. Consecutive time slots constraint
for i in movie_indices:
    for j in range(len(my_channel_df)):
        for k in range(j + 1, len(my_channel_df)):
            model.addConstraint(x[i][j] * my_channel_df['Date-Time'].iloc[k] - x[i][j] * my_channel_df['Date-Time'].iloc[j] <= x[i][j] * movie_db_df['runtime'].iloc[i],
                                f"ConsecutiveSlots_{i}_{j}_{k}")
print('constraint 4 added at time ', time() - start_time)

# 5. Budget constraint for movies
total_budget = 1000000  # Example budget
model.addConstraint(
    xp.Sum(x[i][j] * movie_db_df['budget'].iloc[i] for i in movie_indices for j in range(len(my_channel_df))) <= total_budget,
    "BudgetConstraint"
)
print('constraint 5 added at time ', time() - start_time)

# 6. Advertising budget constraint
total_ad_budget = 500000  # Example advertising budget
model.addConstraint(
    xp.Sum(ad_vars[ch] * (other_channels_0_df['ad_slot_price'].sum() if ch == 'Channel_0' else
                          other_channels_1_df['ad_slot_price'].sum() if ch == 'Channel_1' else
                          other_channels_2_df['ad_slot_price'].sum())
               for ch in ad_indices) <= total_ad_budget,
    "AdBudgetConstraint"
)
print('constraint 6 added at time ', time() - start_time)

# 7. Threshold for Conversion Rates
conversion_rate_threshold = 0.2  # Example threshold
for i in movie_indices:
    model.addConstraint(x[i][j] * (xp.Sum(conversion_rates_mapping[ch].iloc[i][1:] for ch in ad_indices) >= conversion_rate_threshold), 
                                     f"ConversionRateConstraint_{i}")
print('constraint 7 added at time ', time() - start_time)

# 8. Daily Genre Diversity Constraint
max_genres_per_day = 3  # Example limit for genres
for j in range(len(my_channel_df)):
    valid_movies = my_channel_df[my_channel_df['Date-Time'] == my_channel_df['Date-Time'].iloc[j]]
    genre_count = valid_movies['genre'].nunique()  # We can adjust this depending on how each genre is represented
    model.addConstraint(genre_count <= max_genres_per_day, f"GenreDiversityConstraint_{j}")
print('constraint 8 added at time ', time() - start_time)

# 9. Genre Clashes Constraint
competitor_schedules = pd.concat([
    other_channels_0_df[['Date-Time', 'Content Type', 'genre']],
    other_channels_1_df[['Date-Time', 'Content Type', 'genre']],
    other_channels_2_df[['Date-Time', 'Content Type', 'genre']]
], ignore_index=True)
print('constraint 9 added at time ', time() - start_time)

# Filter to keep only the scheduled movies (not advertisements)
competitor_movies = competitor_schedules[competitor_schedules['Content Type'] == 'Movie']

# Add constraints to avoid genre clashes
for i in movie_indices:
    movie_genre = movie_db_df['genre'].iloc[i]  # Get the genre of the movie to be scheduled
    for j in range(len(my_channel_df)):
        if my_channel_df['Date-Time'].iloc[j] is not None:
            # Find all competitor movies scheduled at the same time
            competing_movies = competitor_movies[competitor_movies['Date-Time'] == my_channel_df['Date-Time'].iloc[j]]
            
            # Add constraints to limit genre clashes
            for _, competing_movie in competing_movies.iterrows():
                if competing_movie['genre'] == movie_genre:
                    model.addConstraint(x[i][j] + xp.Sum(x[other_i][j] for other_i in movie_indices if movie_db_df['genre'].iloc[other_i] == movie_genre) <= 1,
                                        f"GenreClashConstraint_{i}_{j}")

print('constraint 10 added at time ', time() - start_time)

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
