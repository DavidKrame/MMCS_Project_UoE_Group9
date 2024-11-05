import pandas as pd
import numpy as np
import xpress as xp
from datetime import datetime, timedelta

xp.init('C:/xpressmp/bin/xpauth.xpr')

my_channel_df = pd.read_csv('data/AGGREGATE_FIRST_WEEK_channel_A_schedule.csv', parse_dates=['Date-Time'])
movie_db_df = pd.read_csv('data/movie_database_with_license_fee_1000.csv', parse_dates=['release_date'])

model = xp.problem()

number_of_movies = len(movie_db_df.index)
number_of_time_slots = len(my_channel_df.index)

Movies = range(number_of_movies)
Time_slots = range(number_of_time_slots)

###################
# Decision Variables
###################

# whether to schedule movie i at time slot j
x = [[xp.var(name=f"x_{i}_{j}", vartype=xp.binary) for j in my_channel_df['Date-Time']] for i in Movies]
model.addVariable([var for sublist in x for var in sublist])

# whether to show movie i
y = [xp.var(name=f"y_{i}", vartype=xp.binary) for i in Movies]
model.addVariable(y)

# start time movie i
s = np.array([xp.var( name='s_{0}'.format(i), vartype=xp.continuous)
                    for i in Movies], dtype=xp.npvar).reshape(number_of_movies)
model.addVariable(s)

# end time movie i
e = np.array([xp.var( name='e_{0}'.format(i), vartype=xp.continuous)
                    for i in Movies], dtype=xp.npvar).reshape(number_of_movies)
model.addVariable(e)

viewership_from_movies = xp.Sum(x[i][j] * movie_db_df['scaled_popularity'].iloc[i] for i in Movies for j in Time_slots)

license_fees = xp.Sum(x[i][j] * movie_db_df['license_fee'].iloc[i] for i in Movies for j in Time_slots)

model.setObjective(viewership_from_movies - license_fees, sense=xp.maximize)

# 1. You can only schedule one movie per time slot
model.addConstraint(xp.Sum(x[i][j] for i in Movies) == 1 for j in Time_slots)

# 2. Movie must be scheduled for whole length
model.addConstraint(30*xp.Sum(x[i][j] for j in Time_slots) == y[i]*movie_db_df['runtime_with_ads'].loc[i] for i in Movies)

# 3. Start and end time must be length of movie
model.addConstraint(e[i] - s[i] == y[i]*movie_db_df['runtime_with_ads'].loc[i] for i in Movies)

# 4. Movie must be scheduled for consecutive time slots
start_of_week = datetime(2024, 10, 1, 0, 0, 0)
end_of_week = datetime(2024, 10, 8, 0, 0, 0)

model.addConstraint(
    e[i] >= x[i][j]*(my_channel_df['Date-Time'].loc[j] - start_of_week).total_seconds()/60
    for i in Movies for j in Time_slots
    )

model.addConstraint(
    s[i] <= x[i][j]*(my_channel_df['Date-Time'].loc[j] - start_of_week).total_seconds()/60 + (1 - x[i][j])*(10080)
    for i in Movies for j in Time_slots
    )

model.controls.maxtime=1000
model.controls.tunermaxtime = 1000
model.controls.timelimit = 60
model.tune('g')
model.solve()

with open("Output1.txt", "w") as f:
    for j in Time_slots:
        for i in Movies:
            if model.getSolution(x[i][j]) == 1:
                f.write("At ")
                f.write(str(my_channel_df['Date-Time'].loc[j]))
                f.write(" show movie ")
                f.write(movie_db_df['title'].loc[1])
                f.write('\n')
f.close()