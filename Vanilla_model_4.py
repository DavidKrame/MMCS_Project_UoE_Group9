import pandas as pd
import numpy as np
import xpress as xp
from datetime import datetime, timedelta
from time import time

import csv
import math

from time_slot_viewership import movie_views_for_time_slot,comp_advertised_views_for_time_slot, own_advertised_views_for_time_slot, calculate_ad_slot_price

start_time = time()

xp.init('C:/xpressmp/bin/xpauth.xpr')

my_channel_df = pd.read_csv('data/AGGREGATE_FIRST_WEEK_channel_A_schedule.csv', parse_dates=['Date-Time'])
movie_db_df = pd.read_csv('data/filtered_movie_database_working_days.csv', parse_dates=['release_date'])
# filtered_movie_database_135
channel_0_df = pd.read_csv('data/ADVERTS_FIRST_WEEK_channel_0_schedule.csv', parse_dates=['Date-Time'])
channel_1_df = pd.read_csv('data/ADVERTS_FIRST_WEEK_channel_1_schedule.csv', parse_dates=['Date-Time'])
channel_2_df = pd.read_csv('data/ADVERTS_FIRST_WEEK_channel_2_schedule.csv', parse_dates=['Date-Time'])
genre_conversion_0_df = pd.read_csv('data/movie_adslots_conversion_0_4block.csv')
genre_conversion_1_df = pd.read_csv('data/movie_adslots_conversion_1_4block.csv')
genre_conversion_2_df = pd.read_csv('data/movie_adslots_conversion_2_4block.csv')


cutoff = datetime(2024, 10, 5, 0, 0, 0)


my_channel_df = my_channel_df.drop(my_channel_df[my_channel_df['Date-Time'] > cutoff].index)
channel_0_df = channel_0_df.drop(channel_0_df[channel_0_df['Date-Time'] > cutoff].index)
channel_1_df = channel_1_df.drop(channel_1_df[channel_1_df['Date-Time'] > cutoff].index)
channel_2_df = channel_2_df.drop(channel_2_df[channel_2_df['Date-Time'] > cutoff].index)

model = xp.problem()

Demos = ['children', 'adults', 'retirees']

number_of_movies = len(movie_db_df.index)
number_of_time_slots = len(my_channel_df.index)
number_of_comp_channels = 3
number_of_ad_slots_0 = len(channel_0_df)
number_of_ad_slots_1 = len(channel_1_df)
number_of_ad_slots_2 = len(channel_2_df)

Movies = range(number_of_movies)
Time_slots = range(number_of_time_slots)
Channels = range(number_of_comp_channels)

Ad_slots_0 = range(number_of_ad_slots_0)
Ad_slots_1 = range(number_of_ad_slots_1)
Ad_slots_2 = range(number_of_ad_slots_2)

population = 1000000
viewership_units = 1000
ad_sell_price_per_unit = 100
budget = 21000000

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

# print('Scheduling variables added, ', time() - start_time)
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

print('Ad slot variables added, ', time() - start_time)
# sum of viewers for movie i across time slots shown
u = np.array(
    [xp.var(name=f"u_{i}_{j}", vartype=xp.continuous) for i in Movies for j in Time_slots]
).reshape(number_of_movies, number_of_time_slots)
model.addVariable(u)

print('Viewership variables added, ', time() - start_time)
# start time movie i
start = np.array([xp.var( name='s_{0}'.format(i), vartype=xp.continuous)
                    for i in Movies], dtype=xp.npvar).reshape(number_of_movies)
model.addVariable(start)

# end time movie i
end = np.array([xp.var( name='e_{0}'.format(i), vartype=xp.continuous)
                    for i in Movies], dtype=xp.npvar).reshape(number_of_movies)
model.addVariable(end)

print('Variables added, ', time() - start_time)
##########################
# Constraints
##########################

# 1. You can only schedule one movie per time slot
model.addConstraint(xp.Sum(x[i][j] for i in Movies) == 1 for j in Time_slots)

print('Constaint 1 added, ', time() - start_time)
# 2. Movie must be scheduled for whole length
model.addConstraint(30*xp.Sum(x[i][j] for j in Time_slots) == y[i]*movie_db_df['runtime_with_ads'].loc[i] for i in Movies)

print('Constaint 2 added, ', time() - start_time)
# 3. Start and end time must be length of movie
model.addConstraint(end[i] - start[i] == y[i]*movie_db_df['runtime_with_ads'].loc[i] for i in Movies)

print('Constaint 3 added, ', time() - start_time)
# 4. Movie must be scheduled for consecutive time slots
start_of_week = datetime(2024, 10, 1, 0, 0, 0)

model.addConstraint(
    end[i] >= x[i][j]*(my_channel_df['Date-Time'].loc[j] + timedelta(minutes=30)- start_of_week).total_seconds()/60
    for i in Movies for j in Time_slots
    )

model.addConstraint(
    start[i] <= x[i][j]*(my_channel_df['Date-Time'].loc[j] - start_of_week).total_seconds()/60 + (1 - x[i][j])*(10080 - movie_db_df['runtime_with_ads'].loc[i])
    for i in Movies for j in Time_slots
    )


print('Constaint 4 added, ', time() - start_time)
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

print('Constaint 5 added, ', time() - start_time)

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

print('Constaint 6 added, ', time() - start_time)

# 7. Only sell ad slot for movie i at time j if it is shown then
model.addConstraint(
    v[i][j] <= x[i][j] for i in Movies for j in Time_slots
)

print('Constaint 7 added, ', time() - start_time)
# 8. Only advertise before the movie is scheduled
model.addConstraint(
    z0[i][r]*(channel_0_df['Date-Time'].iloc[r] - start_of_week).total_seconds()/60 <= start[i]
    for i in Movies for r in Ad_slots_0
)
model.addConstraint(
    z1[i][s]*(channel_1_df['Date-Time'].iloc[s] - start_of_week).total_seconds()/60 <= start[i]
    for i in Movies for s in Ad_slots_1
)
model.addConstraint(
    z2[i][t]*(channel_2_df['Date-Time'].iloc[t] - start_of_week).total_seconds()/60 <= start[i]
    for i in Movies for t in Ad_slots_2
)
model.addConstraint(
    w[i][j]*(my_channel_df['Date-Time'].iloc[j] - start_of_week + timedelta(minutes=30)).total_seconds()/60 <= start[i]
    for i in Movies for j in Time_slots
)

print('Constaint 8 added, ', time() - start_time)

model.addConstraint(
    u[i][j]
    <=
    movie_views_for_time_slot(x, i, j, movie_db_df, my_channel_df, Demos, population)
    + xp.Sum(
        own_advertised_views_for_time_slot(w, i , k, movie_db_df, my_channel_df, Demos, population)
        for k in Time_slots
    )
    + xp.Sum(
        comp_advertised_views_for_time_slot(z0, i, r, movie_db_df, channel_0_df, Demos, population, genre_conversion_0_df)
        for r in Ad_slots_0
    )
    + xp.Sum(
        comp_advertised_views_for_time_slot(z1, i, s, movie_db_df, channel_1_df, Demos, population, genre_conversion_1_df)
        for s in Ad_slots_1
    )
    + xp.Sum(
        comp_advertised_views_for_time_slot(z2, i, t, movie_db_df, channel_2_df, Demos, population, genre_conversion_2_df)
        for t in Ad_slots_2
    )
    for i in Movies for j in Time_slots
)

print('Constaint u added, ', time() - start_time)
# 10. We only get contribution for viewership for movie i at time slot j if the time slot is sold
model.addConstraint(
    u[i][j] <= v[i][j]*(population)
    for i in Movies for j in Time_slots
)


print('Constaint 10 added, ', time() - start_time)

# 11. license fees and advertising slots bought must be within budget
model.addConstraint(
    xp.Sum(
        y[i] * movie_db_df['license_fee'].iloc[i]
        for i in Movies
    )
    + xp.Sum(
        z0[i][r] * channel_0_df['ad_slot_price'].iloc[r]
        for i in Movies for r in Ad_slots_0
    )
    + xp.Sum(
        z1[i][s] * channel_1_df['ad_slot_price'].iloc[s]
        for i in Movies for s in Ad_slots_1
    )
    + xp.Sum(
        z2[i][t] * channel_2_df['ad_slot_price'].iloc[t]
        for i in Movies for t in Ad_slots_2
    )
    <= budget
)

print('Constaint 11 added, ', time() - start_time)

##########################
# Objective Function
##########################

model.setObjective(
    xp.Sum(u[i][j] for i in Movies for j in Time_slots),
    sense=xp.maximize
)

print('time to intialise problem: ', time() - start_time)

model.controls.maxtime = 120
model.controls.heurfreq = -1  # Disable heuristic frequency
model.controls.heuremphasis = 0  # No heuristics
# model.controls.maxnode = 1000
# model.controls.miprelstop = 0.01
# model.controls.tunermaxtime = 1000
# model.controls.timelimit = 60
# model.tune('g')

solvestatus, solstatus = model.optimize()

print('solve time, ', time() - start_time)

now = datetime.now()
now = str(now).replace(" ", "_")
now = now.replace(":", "-")

# saved_sol_path = f'solutions/scheduling_advert_demos_{now}'
# model.write(saved_sol_path)

# model.iisfirst(1)

# rowind = []
# colind = []
# contype = []
# bndtype = []
# duals = []
# djs = []
# isolationrows = []
# isolationcols = []
# model.getiisdata(1, rowind, colind, contype, bndtype,
#               duals, djs, isolationrows, isolationcols)

x_sol = model.getSolution(x)
y_sol = model.getSolution(y)
z0_sol = model.getSolution(z0)
z1_sol = model.getSolution(z1)
z2_sol = model.getSolution(z2)
w_sol = model.getSolution(w)
v_sol = model.getSolution(v)
u_sol = model.getSolution(u)
# q_sol = model.getSolution(q)

# pd.DataFrame(x_sol).to_csv(f'solutions/x_sol_{now}.csv')
# pd.DataFrame(y_sol).to_csv(f'solutions/y_sol_{now}.csv')
# pd.DataFrame(z0_sol).to_csv(f'solutions/z0_sol_{now}.csv')
# pd.DataFrame(z1_sol).to_csv(f'solutions/z1_sol_{now}.csv')
# pd.DataFrame(z2_sol).to_csv(f'solutions/z2_sol_{now}.csv')
# pd.DataFrame(w_sol).to_csv(f'solutions/w_sol_{now}.csv')
# pd.DataFrame(v_sol).to_csv(f'solutions/v_sol_{now}.csv')
# pd.DataFrame(q_sol).to_csv(f'solutions/q_sol_{now}.csv')

cost = sum(y_sol[i] * movie_db_df['license_fee'].iloc[i] for i in Movies)
+ sum(z0_sol[i][r] * channel_0_df['ad_slot_price'].loc[r] for i in Movies for r in Ad_slots_0)
+ sum(z1_sol[i][s] * channel_1_df['ad_slot_price'].loc[s] for i in Movies for s in Ad_slots_1)
+ sum(z2_sol[i][t] * channel_2_df['ad_slot_price'].loc[t] for i in Movies for t in Ad_slots_2)
print(cost)
# # if solstatus != xp.SolStatus.INFEASIBLE or solstatus != xp.SolStatus.UNBOUNDED or solstatus != xp.SolStatus.UNBOUNDED:
with open(f"./output/output_NO_HEURISTICS_222_7Days_100Movies_{str(now)}.txt", "w") as f:
    # f.write('Viewership: ')
    # f.write(str(model.getObjVal()))
    # f.write('\n')
    f.write('Total cost: ')
    f.write(str(cost))
    f.write('\n')
    for j in Time_slots:
        for i in Movies:
            if x_sol[i][j] == 1:
                f.write("At ")
                f.write(str(my_channel_df['Date-Time'].loc[j]))
                f.write(" show movie ")
                f.write(movie_db_df['title'].loc[i])
        for i in Movies:
            if w_sol[i][j] == 1:
                f.write(", on own channel advertise movie ")
                f.write(movie_db_df['title'].loc[i])
        for i in Movies:
            if v_sol[i][j] == 1:
                f.write(", sell adslot with viewership ")
                f.write(str(u_sol[i][j]))
        f.write('\n')
    for j in Ad_slots_0:
        for i in Movies:
            if z0_sol[i][j] == 1:
                f.write("At ")
                f.write(str(channel_0_df['Date-Time'].loc[j]))
                f.write(" on channel 0 advertise movie ")
                f.write(movie_db_df['title'].loc[i])
                f.write('\n')
    for j in Ad_slots_1:
        for i in Movies:
            if z1_sol[i][j] == 1:
                f.write("At ")
                f.write(str(channel_1_df['Date-Time'].loc[j]))
                f.write(" on channel 1 advertise movie ")
                f.write(movie_db_df['title'].loc[i])
                f.write('\n')
    for j in Ad_slots_2:
        for i in Movies:
            if z2_sol[i][j] == 1:
                f.write("At ")
                f.write(str(channel_2_df['Date-Time'].loc[j]))
                f.write(" on channel 2 advertise movie ")
                f.write(movie_db_df['title'].loc[i])
                f.write('\n')
f.close()

print('solution output, ', time() - start_time)

# STORE IN A CSV FILE, FIRST TRIAL

output_filename = f"./output/output_4Days_{str(now)}_budget_{budget}.csv"

csv_data = []
csv_data.append(['ID', 'Date-Time', 'Channel', 'Action', 'Movie Title', 'Viewerships'])

# Total cost just recorded this way
csv_data.append(['Cost', 'N/A', 'N/A', 'Total Cost', 'N/A', str(cost)])

# Process own channel stuffs
for j in Time_slots:
    for i in Movies:
        date_time = my_channel_df['Date-Time'].loc[j]
        if x_sol[i][j] == 1:
            action = 'Show Movie'
            channel = 'Own Channel'
            movie_title = movie_db_df['title'].loc[i]
            viewership = ''
            csv_data.append([f'{date_time}_{channel}_{action}', date_time, channel, action, movie_title, viewership])
        if w_sol[i][j] == 1:
            action = 'Advertise Movie'
            channel = 'Own Channel'
            movie_title = movie_db_df['title'].loc[i]
            viewership = ''
            csv_data.append([f'{date_time}_{channel}_{action}', date_time, channel, action, movie_title, viewership])
        if v_sol[i][j] == 1:
            action = 'Sell Adslot'
            channel = 'Own Channel'
            movie_title = ''
            viewership = math.ceil(u_sol[i][j])
            csv_data.append([f'{date_time}_{channel}_{action}', date_time, channel, action, movie_title, viewership])

# competitor channel events
def process_competitor_channel(channel, ad_slots, z_sol, channel_df):
    for j in ad_slots:
        for i in Movies:
            if z_sol[i][j] == 1:
                date_time = channel_df['Date-Time'].loc[j]
                action = 'Advertise Movie'
                movie_title = movie_db_df['title'].loc[i]
                viewership = ''
                csv_data.append([f'{date_time}_{channel}_{action}', date_time, channel, action, movie_title, viewership])

process_competitor_channel('Channel 0', Ad_slots_0, z0_sol, channel_0_df)
process_competitor_channel('Channel 1', Ad_slots_1, z1_sol, channel_1_df)
process_competitor_channel('Channel 2', Ad_slots_2, z2_sol, channel_2_df)

# sort data by Date-Time
csv_data = csv_data[:1] + sorted(csv_data[1:], key=lambda row: (row[1] if row[1] != 'N/A' else ''))

with open(output_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print(f'Solution output written to {output_filename}, Time: {time() - start_time}')
