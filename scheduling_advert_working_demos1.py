import pandas as pd
import numpy as np
import xpress as xp
from datetime import datetime, timedelta
from time import time

start =  time()
xp.init('C:/xpressmp/bin/xpauth.xpr')

my_channel_df = pd.read_csv('data/AGGREGATE_FIRST_WEEK_channel_A_schedule.csv', parse_dates=['Date-Time'])
movie_db_df = pd.read_csv('data/movie_database_with_license_fee_50.csv', parse_dates=['release_date'])
movie_genre_df = pd.read_csv('data/movie_genre_hot_one_50.csv')


channel_0_df = pd.read_csv('data/AGGREGATE_FIRST_WEEK_channel_0_schedule.csv', parse_dates=['Date-Time'])
channel_1_df = pd.read_csv('data/AGGREGATE_FIRST_WEEK_channel_1_schedule.csv', parse_dates=['Date-Time'])
channel_2_df = pd.read_csv('data/AGGREGATE_FIRST_WEEK_channel_2_schedule.csv', parse_dates=['Date-Time'])
conversion_rates_0_df = pd.read_csv('data/AGGREGATE_FIRST_WEEK_channel_0_conversion_rates.csv', parse_dates=['Date-Time'])
conversion_rates_1_df = pd.read_csv('data/AGGREGATE_FIRST_WEEK_channel_1_conversion_rates.csv', parse_dates=['Date-Time'])
conversion_rates_2_df = pd.read_csv('data/AGGREGATE_FIRST_WEEK_channel_2_conversion_rates.csv', parse_dates=['Date-Time'])


first_week_cutoff = datetime(2024, 10, 2, 23, 59, 0)
my_channel_df = my_channel_df.drop(my_channel_df[my_channel_df['Date-Time'] > first_week_cutoff].index)

model = xp.problem()

Demos = ['children', 'adults', 'retirees']
Genres = movie_genre_df.columns.to_list()

number_of_movies = len(movie_db_df.index)
number_of_time_slots = len(my_channel_df.index)
number_of_comp_channels = 3

Movies = range(number_of_movies)
Time_slots = range(number_of_time_slots)
Channels = range(number_of_comp_channels)

channel_dict = {
    0: channel_0_df,
    1: channel_1_df,
    2: channel_2_df
}

conversion_dict = {
    0: conversion_rates_0_df,
    1: conversion_rates_1_df,
    2: conversion_rates_2_df
}

population = 1000000

####################
# Decision Variables
####################

# whether to schedule movie i at time slot j
x = [[xp.var(name=f"x_{i}_{j}", vartype=xp.binary) for j in Time_slots] for i in Movies]
model.addVariable([var for sublist in x for var in sublist])

# whether to show movie i
y = [xp.var(name=f"y_{i}", vartype=xp.binary) for i in Movies]
model.addVariable(y)

# whether movie i is advertises on channel c at time slot j
z = np.array(
    [xp.var(name=f'z_{i}_{j}_{c}', vartype=xp.binary) for i in Movies for j in Time_slots for c in Channels],
    dtype=xp.npvar
    ).reshape(number_of_movies,number_of_time_slots, number_of_comp_channels)

model.addVariable(z)

# start time movie i
s = np.array([xp.var( name='s_{0}'.format(i), vartype=xp.continuous)
                    for i in Movies], dtype=xp.npvar).reshape(number_of_movies)
model.addVariable(s)

# end time movie i
e = np.array([xp.var( name='e_{0}'.format(i), vartype=xp.continuous)
                    for i in Movies], dtype=xp.npvar).reshape(number_of_movies)
model.addVariable(e)

viewership_from_movies = xp.Sum(
    x[i][j] * xp.Sum(
        (population*movie_db_df[f'{k}_scaled_popularity'].iloc[i])
        *my_channel_df[f'{k}_baseline_view_count'].iloc[j]
        for k in Demos)
    for i in Movies for j in Time_slots
    )

print(time() - start, ' before viewership_from_bought_ads')
viewership_from_bought_ads= xp.Sum(
    z[i][j][c] * xp.Sum(
        (population * movie_db_df[f'{k}_scaled_popularity'].iloc[i])
        *channel_dict[c][f'{k}_baseline_view_count'].loc[j]
        *conversion_dict[c][g].loc[i]
        for g in Genres for k in Demos
    )
    for i in Movies for j in Time_slots for c in Channels
)


license_fees = xp.Sum(x[i][j] * movie_db_df['license_fee'].iloc[i] for i in Movies for j in Time_slots)

model.setObjective(
    viewership_from_movies + viewership_from_bought_ads,
    sense=xp.maximize
    )

# 1. You can only schedule one movie per time slot
model.addConstraint(xp.Sum(x[i][j] for i in Movies) == 1 for j in Time_slots)

# 2. Movie must be scheduled for whole length
model.addConstraint(30*xp.Sum(x[i][j] for j in Time_slots) == y[i]*movie_db_df['runtime_with_ads'].loc[i] for i in Movies)

# 3. Start and end time must be length of movie
model.addConstraint(e[i] - s[i] == y[i]*movie_db_df['runtime_with_ads'].loc[i] for i in Movies)

# 4. Movie must be scheduled for consecutive time slots
start_of_week = datetime(2024, 10, 1, 0, 0, 0)

model.addConstraint(
    e[i] >= x[i][j]*(my_channel_df['Date-Time'].loc[j] + timedelta(minutes=30)- start_of_week).total_seconds()/60
    for i in Movies for j in Time_slots
    )

model.addConstraint(
    s[i] <= x[i][j]*(my_channel_df['Date-Time'].loc[j] - start_of_week).total_seconds()/60 + (1 - x[i][j])*(2880 - movie_db_df['runtime_with_ads'].loc[i])
    for i in Movies for j in Time_slots
    )

# 5. Only one ad can be bought per avialable slot
model.addConstraint(
    xp.Sum(z[i][j][c] for i in Movies) == 1 for j in Time_slots for c in Channels
)

# 6. Only advertise movie if it is shown
model.addConstraint(
    z[i][j][c] <= y[i] for i in Movies for j in Time_slots for c in Channels
)

# 7. Only advertise before the movie is scheduled
model.addConstraint(
    z[i][j][c]*(my_channel_df['Date-Time'].loc[i] - start_of_week).total_seconds()/60 <= s[i] - 30
    for i in Movies for j in Time_slots for c in Channels
)

# model.controls.maxtime = 300
model.controls.maxnode = 1000  # Limit to 1000 nodes
model.controls.miprelstop = 0.01  # Stop if relative gap is less than 1%
# model.controls.tunermaxtime = 1000
# model.controls.timelimit = 60
# model.tune('g')

solvestatus, solstatus = model.optimize()

now = datetime.now()
now = str(now).replace(" ", "_")
now = now.replace(":", "-")

saved_sol_path = f'solutions/scheduling_advert_demos_{now}'
model.write(saved_sol_path)

if solstatus != xp.SolStatus.INFEASIBLE or solstatus != xp.SolStatus.UNBOUNDED or solstatus != xp.SolStatus.UNBOUNDED:
    with open(f"./output/output_{str(now)}.txt", "w") as f:
        f.write('Viewership: ')
        f.write(str(model.getObjVal))
        f.write('\n')
        for j in Time_slots:
            for i in Movies:
                if model.getSolution(x[i][j]) == 1:
                    f.write("At ")
                    f.write(str(my_channel_df['Date-Time'].loc[j]))
                    f.write(" show movie ")
                    f.write(movie_db_df['title'].loc[i])
                    f.write('\n')
    f.close()
