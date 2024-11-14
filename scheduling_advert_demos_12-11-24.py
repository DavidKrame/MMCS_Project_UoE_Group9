import pandas as pd
import numpy as np
import xpress as xp
from datetime import datetime, timedelta
from time import time

from time_slot_viewership import movie_views_for_time_slot,comp_advertised_views_for_time_slot, own_advertised_views_for_time_slot, calculate_ad_slot_price

start_time = time()
time_stamp = 0

xp.init('C:/xpressmp/bin/xpauth.xpr')

my_channel_df = pd.read_csv('data/AGGREGATE_FIRST_WEEK_channel_A_schedule.csv', parse_dates=['Date-Time'])
movie_db_df = pd.read_csv('data/movie_database_with_license_fee_100.csv', parse_dates=['release_date'])
movie_genre_df = pd.read_csv('data/movie_genre_hot_one_100.csv')


channel_0_df = pd.read_csv('data/ADVERTS_FIRST_WEEK_channel_0_schedule.csv', parse_dates=['Date-Time'])
channel_1_df = pd.read_csv('data/ADVERTS_FIRST_WEEK_channel_0_schedule.csv', parse_dates=['Date-Time'])
channel_2_df = pd.read_csv('data/ADVERTS_FIRST_WEEK_channel_0_schedule.csv', parse_dates=['Date-Time'])
conversion_rates_0_df = pd.read_csv('data/FIRST_WEEK_channel_0_conversion_rates.csv', parse_dates=['Date-Time'])
conversion_rates_1_df = pd.read_csv('data/FIRST_WEEK_channel_1_conversion_rates.csv', parse_dates=['Date-Time'])
conversion_rates_2_df = pd.read_csv('data/FIRST_WEEK_channel_2_conversion_rates.csv', parse_dates=['Date-Time'])


first_week_cutoff = datetime(2024, 10, 2, 23, 59, 0)
my_channel_df = my_channel_df.drop(my_channel_df[my_channel_df['Date-Time'] > first_week_cutoff].index)
channel_0_df = channel_0_df.drop(channel_0_df[channel_0_df['Date-Time'] > first_week_cutoff].index)
channel_1_df = channel_1_df.drop(channel_1_df[channel_1_df['Date-Time'] > first_week_cutoff].index)
channel_2_df = channel_2_df.drop(channel_2_df[channel_2_df['Date-Time'] > first_week_cutoff].index)
conversion_rates_0_df = conversion_rates_0_df.drop(conversion_rates_0_df[conversion_rates_0_df['Date-Time'] > first_week_cutoff].index)
conversion_rates_1_df = conversion_rates_1_df.drop(conversion_rates_1_df[conversion_rates_1_df['Date-Time'] > first_week_cutoff].index)
conversion_rates_2_df = conversion_rates_2_df.drop(conversion_rates_2_df[conversion_rates_2_df['Date-Time'] > first_week_cutoff].index)

model = xp.problem()

Demos = ['children', 'adults', 'retirees']
Genres = movie_genre_df.columns.to_list()

number_of_movies = len(movie_db_df.index)
number_of_time_slots = len(my_channel_df.index)
number_of_comp_channels = 3
number_of_ad_slots_0 = len(conversion_rates_0_df)
number_of_ad_slots_1 = len(conversion_rates_1_df)
number_of_ad_slots_2 = len(conversion_rates_2_df)

Movies = range(number_of_movies)
Time_slots = range(number_of_time_slots)
Channels = range(number_of_comp_channels)

Ad_slots_0 = range(number_of_ad_slots_0)
Ad_slots_1 = range(number_of_ad_slots_1)
Ad_slots_2 = range(number_of_ad_slots_2)

population = 1000000
viewership_units = 1000
ad_sell_price_per_unit = 100
budget = 6000000

##########################
# Decision Variables
##########################

# whether to schedule movie i at time slot j
x = np.array(
    [xp.var(name=f"x_{i}_{j}", vartype=xp.binary) for i in Movies for j in Time_slots],
    dtype=xp.npvar
).reshape(number_of_movies,number_of_time_slots)
model.addVariable(x)

# whether to show movie i
y = [xp.var(name=f"y_{i}", vartype=xp.binary) for i in Movies]
model.addVariable(y)

# whether movie i is advertises on channel 0 at time slot j
z0 = np.array(
    [xp.var(name=f'z0_{i}_{r}', vartype=xp.binary) for i in Movies for r in Ad_slots_0],
    dtype=xp.npvar
).reshape(number_of_movies,number_of_ad_slots_0)
model.addVariable(z0)

# whether movie i is advertises on channel 0 at time slot j
z1 = np.array(
    [xp.var(name=f'z1_{i}_{s}', vartype=xp.binary) for i in Movies for s in Ad_slots_1],
    dtype=xp.npvar
).reshape(number_of_movies,number_of_ad_slots_1)
model.addVariable(z1)

# whether movie i is advertises on channel 0 at time slot j
z2 = np.array(
    [xp.var(name=f'z2_{i}_{t}', vartype=xp.binary) for i in Movies for t in Ad_slots_2],
    dtype=xp.npvar
).reshape(number_of_movies,number_of_ad_slots_2)
model.addVariable(z2)

# whether movie i is advertised on our own channel at time slot j
w = np.array(
    [xp.var(name=f"w_{i}_{j}", vartype=xp.binary) for i in Movies for j in Time_slots],
    dtype=xp.npvar
).reshape(number_of_movies,number_of_time_slots)
model.addVariable(w)

# whether advert slot at time slot j is sold
v = np.array(
    [xp.var(name=f"v_{i}_{j}", vartype=xp.binary) for i in Movies for j in Time_slots]
).reshape(number_of_movies, number_of_time_slots)
model.addVariable(v)

# sum of viewers for movie i across time slots shown
u = np.array(
    [xp.var(name=f"u_{i}_{j}", vartype=xp.continuous) for i in Movies for j in Time_slots]
).reshape(number_of_movies, number_of_time_slots)
model.addVariable(u)

uA = [xp.var(name=f"uA_{i}", vartype=xp.continuous) for i in Movies]
model.addVariable(uA)

u0 = [xp.var(name=f"u0_{i}", vartype=xp.continuous) for i in Movies]
model.addVariable(u0)

u1= [xp.var(name=f"u1_{i}", vartype=xp.continuous) for i in Movies]
model.addVariable(u1)

u2 = [xp.var(name=f"u2_{i}", vartype=xp.continuous) for i in Movies]
model.addVariable(u2)

q = np.array(
    [xp.var(name=f"q_{i}_{j}", vartype=xp.continuous) for i in Movies for j in Time_slots]
).reshape(number_of_movies,number_of_time_slots)
model.addVariable(q)

# start time movie i
start = np.array([xp.var( name='s_{0}'.format(i), vartype=xp.continuous)
                    for i in Movies], dtype=xp.npvar).reshape(number_of_movies)
model.addVariable(start)

# end time movie i
end = np.array([xp.var( name='e_{0}'.format(i), vartype=xp.continuous)
                    for i in Movies], dtype=xp.npvar).reshape(number_of_movies)
model.addVariable(end)

##########################
# Constraints
##########################

# 1. You can only schedule one movie per time slot
model.addConstraint(xp.Sum(x[i][j] for i in Movies) == 1 for j in Time_slots)

# 2. Movie must be scheduled for whole length
model.addConstraint(30*xp.Sum(x[i][j] for j in Time_slots) == y[i]*movie_db_df['runtime_with_ads'].loc[i] for i in Movies)

# 3. Start and end time must be length of movie
model.addConstraint(end[i] - start[i] == y[i]*movie_db_df['runtime_with_ads'].loc[i] for i in Movies)

# 4. Movie must be scheduled for consecutive time slots
start_of_week = datetime(2024, 10, 1, 0, 0, 0)

model.addConstraint(
    end[i] >= x[i][j]*(my_channel_df['Date-Time'].loc[j] + timedelta(minutes=30)- start_of_week).total_seconds()/60
    for i in Movies for j in Time_slots
    )

model.addConstraint(
    start[i] <= x[i][j]*(my_channel_df['Date-Time'].loc[j] - start_of_week).total_seconds()/60 + (1 - x[i][j])*(2880 - movie_db_df['runtime_with_ads'].loc[i])
    for i in Movies for j in Time_slots
    )

print('Scheduling constaints added, ', time() - start_time)
# 5. Only one ad can be bought per avialable slot
model.addConstraint(
    xp.Sum(z0[i][r] for i in Movies) <= 1 for r in Ad_slots_0
)

model.addConstraint(
    xp.Sum(z1[i][s] for i in Movies) <= 1 for s in Ad_slots_1
)

model.addConstraint(
    xp.Sum(z2[i][t] for i in Movies) <= 1 for t in Ad_slots_2
)

model.addConstraint(
    xp.Sum(w[i][j] for i in Movies) + xp.Sum(v[i][j] for i in Movies) == 1 for j in Time_slots
)


# 6. Only advertise movie if it is shown
model.addConstraint(
    z0[i][r] <= y[i] for i in Movies for r in Ad_slots_0
)
model.addConstraint(
    z1[i][s] <= y[i] for i in Movies for s in Ad_slots_1
)
model.addConstraint(
    z2[i][t] <= y[i] for i in Movies for t in Ad_slots_2
)
model.addConstraint(
    w[i][j] <= y[i] for i in Movies for j in Time_slots
)

# 7. Only sell ad slot for movie i at time j if it is shown then
model.addConstraint(
    v[i][j] <= x[i][j] for i in Movies for j in Time_slots
)

# 8. Only advertise before the movie is scheduled
model.addConstraint(
    z0[i][r]*(conversion_rates_0_df['Date-Time'].loc[r] - start_of_week + timedelta(minutes=30)).total_seconds()/60 <= start[i]
    for i in Movies for r in Ad_slots_0
)
model.addConstraint(
    z1[i][s]*(conversion_rates_1_df['Date-Time'].loc[s] - start_of_week + timedelta(minutes=30)).total_seconds()/60 <= start[i]
    for i in Movies for s in Ad_slots_1
)
model.addConstraint(
    z2[i][t]*(conversion_rates_2_df['Date-Time'].loc[t] - start_of_week + timedelta(minutes=30)).total_seconds()/60 <= start[i]
    for i in Movies for t in Ad_slots_2
)
model.addConstraint(
    w[i][j]*(my_channel_df['Date-Time'].loc[j] - start_of_week + timedelta(minutes=30)).total_seconds()/60 <= start[i]
    for i in Movies for j in Time_slots
)

print('Advertising constraints added, ', time() - start_time)
# 9. The number per thousand of viewership is less than the viewership for the time slot
model.addConstraint(
    u[i][j]<=
    movie_views_for_time_slot(x, i, j, movie_db_df, my_channel_df, Demos, population)
    for i in Movies for j in Time_slots
)

model.addConstraint(
    uA[i] <=
    xp.Sum(
        own_advertised_views_for_time_slot(w, i , j, movie_db_df, my_channel_df, Demos, population)
        for j in Time_slots
    )
    for i in Movies
)

model.addConstraint(
    u0[i] <=
    xp.Sum(
        comp_advertised_views_for_time_slot(z0, i, r, movie_db_df, channel_0_df, conversion_rates_0_df, Demos, Genres, population)
        for r in Ad_slots_0
    )
    for i in Movies
)

model.addConstraint(
    u1[i] <=
    xp.Sum(
        comp_advertised_views_for_time_slot(z1, i, s, movie_db_df, channel_1_df, conversion_rates_1_df, Demos, Genres, population)
        for s in Ad_slots_1
    )
    for i in Movies
)

model.addConstraint(
    u2[i] <=
    xp.Sum(
        comp_advertised_views_for_time_slot(z2, i, t, movie_db_df, channel_2_df, conversion_rates_2_df, Demos, Genres, population)
        for t in Ad_slots_2
    )
    for i in Movies
)


# 12. We only get contribution for viewership for movie i at time slot j if the time slot is sold
model.addConstraint(
    q[i][j] <= v[i][j]*(population)
    for i in Movies for j in Time_slots
)

model.addConstraint(
    q[i][j] == u[i][j] + uA[i] + u0[i] + u1[i] + u2[i]
    for i in Movies for j in Time_slots
)

print('Viewership constaints added, ', time() - start_time)

# 12. license fees and advertising slots bought must be within budget
model.addConstraint(
    xp.Sum(
        y[i] * movie_db_df['license_fee'].iloc[i]
        for i in Movies
    )
    + xp.Sum(
        z0[i][r] * channel_0_df['ad_slot_price'].loc[r]
        for i in Movies for r in Ad_slots_0
    )
    + xp.Sum(
        z1[i][s] * channel_1_df['ad_slot_price'].loc[s]
        for i in Movies for s in Ad_slots_1
    )
    + xp.Sum(
        z2[i][t] * channel_2_df['ad_slot_price'].loc[t]
        for i in Movies for t in Ad_slots_2
    )
    <= budget
)


##########################
# Objective Function
##########################

model.setObjective(
    xp.Sum(q[i][j] for i in Movies for j in Time_slots),
    sense=xp.maximize
)

print('time to intialise problem: ', time() - start_time)

# model.controls.maxtime = 300
# model.controls.maxnode = 1000
model.controls.miprelstop = 0.01
# model.controls.tunermaxtime = 1000
# model.controls.timelimit = 60
# model.tune('g')

solvestatus, solstatus = model.optimize()

now = datetime.now()
now = str(now).replace(" ", "_")
now = now.replace(":", "-")

# saved_sol_path = f'solutions/scheduling_advert_demos_{now}'
# model.write(saved_sol_path)

cost = sum(model.getSolution(y[i]) * movie_db_df['license_fee'].iloc[i] for i in Movies)
+ sum(model.getSolution(z0[i][r]) * channel_0_df['ad_slot_price'].loc[r] for i in Movies for r in Ad_slots_0)
+ sum(model.getSolution(z1[i][s]) * channel_1_df['ad_slot_price'].loc[s] for i in Movies for s in Ad_slots_1)
+ sum(model.getSolution(z2[i][t]) * channel_2_df['ad_slot_price'].loc[t] for i in Movies for t in Ad_slots_2)
print(cost)
# if solstatus != xp.SolStatus.INFEASIBLE or solstatus != xp.SolStatus.UNBOUNDED or solstatus != xp.SolStatus.UNBOUNDED:
with open(f"./output/output_{str(now)}.txt", "w") as f:
    # f.write('Viewership: ')
    # f.write(str(model.getObjVal()))
    # f.write('\n')
    f.write('Total cost: ')
    f.write(str(cost))
    f.write('\n')
    for j in Time_slots:
        for i in Movies:
            if model.getSolution(x[i][j]) == 1:
                f.write("At ")
                f.write(str(my_channel_df['Date-Time'].loc[j]))
                f.write(" show movie ")
                f.write(movie_db_df['title'].loc[i])
        for i in Movies:
            if model.getSolution(w[i][j]) == 1:
                f.write(", on own channel advertise movie ")
                f.write(movie_db_df['title'].loc[i])
        for i in Movies:
            if model.getSolution(v[i][j]) == 1:
                f.write(", sell adslot with viewership ")
                f.write(str(model.getSolution(q[i][j])))
        f.write('\n')
    for j in Ad_slots_0:
        for i in Movies:
            if model.getSolution(z0[i][j]) == 1:
                f.write("At ")
                f.write(str(channel_0_df['Date-Time'].loc[j]))
                f.write(" on channel 0 advertise movie ")
                f.write(movie_db_df['title'].loc[i])
                f.write('\n')
    for j in Ad_slots_1:
        for i in Movies:
            if model.getSolution(z1[i][j]) == 1:
                f.write("At ")
                f.write(str(channel_1_df['Date-Time'].loc[j]))
                f.write(" on channel 1 advertise movie ")
                f.write(movie_db_df['title'].loc[i])
                f.write('\n')
    for j in Ad_slots_2:
        for i in Movies:
            if model.getSolution(z2[i][j]) == 1:
                f.write("At ")
                f.write(str(channel_2_df['Date-Time'].loc[j]))
                f.write(" on channel 2 advertise movie ")
                f.write(movie_db_df['title'].loc[i])
                f.write('\n')
f.close()
