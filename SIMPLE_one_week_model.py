import pandas as pd
import numpy as np
import xpress as xp
from datetime import datetime, timedelta
from time import time

from time_slot_viewership import movie_views_for_time_slot,comp_advertised_views_for_time_slot, own_advertised_views_for_time_slot, calculate_ad_slot_price

start_time = time()

xp.init('C:/xpressmp/bin/xpauth.xpr')

my_channel_df = pd.read_csv('data/AGGREGATE_FIRST_WEEK_channel_A_schedule.csv', parse_dates=['Date-Time'])
movie_db_df = pd.read_csv('data/filtered_movie_database.csv', parse_dates=['release_date'])
one_hot_genre_movie_advertized_df = pd.read_csv("data/filtered_movie_db_one_hot_genres.csv")


channel_0_df = pd.read_csv('data/ADVERTS_FIRST_WEEK_channel_0_schedule.csv', parse_dates=['Date-Time'])
channel_1_df = pd.read_csv('data/ADVERTS_FIRST_WEEK_channel_1_schedule.csv', parse_dates=['Date-Time'])
channel_2_df = pd.read_csv('data/ADVERTS_FIRST_WEEK_channel_2_schedule.csv', parse_dates=['Date-Time'])
conversion_rates_0_df = pd.read_csv('data/FIRST_WEEK_channel_0_conversion_rates.csv', parse_dates=['Date-Time'])
conversion_rates_1_df = pd.read_csv('data/FIRST_WEEK_channel_1_conversion_rates.csv', parse_dates=['Date-Time'])
conversion_rates_2_df = pd.read_csv('data/FIRST_WEEK_channel_2_conversion_rates.csv', parse_dates=['Date-Time'])
genre_conversion_0_df = pd.read_csv('data/movies_adslots_conversion_0.csv')
genre_conversion_1_df = pd.read_csv('data/movies_adslots_conversion_1.csv')
genre_conversion_2_df = pd.read_csv('data/movies_adslots_conversion_2.csv')

first_week_cutoff = datetime(2024, 10, 1, 23, 59, 0)
my_channel_df = my_channel_df.drop(my_channel_df[my_channel_df['Date-Time'] > first_week_cutoff].index)
channel_0_df = channel_0_df.drop(channel_0_df[channel_0_df['Date-Time'] > first_week_cutoff].index)
channel_1_df = channel_1_df.drop(channel_1_df[channel_1_df['Date-Time'] > first_week_cutoff].index)
channel_2_df = channel_2_df.drop(channel_2_df[channel_2_df['Date-Time'] > first_week_cutoff].index)
conversion_rates_0_df = conversion_rates_0_df.drop(conversion_rates_0_df[conversion_rates_0_df['Date-Time'] > first_week_cutoff].index)
conversion_rates_1_df = conversion_rates_1_df.drop(conversion_rates_1_df[conversion_rates_1_df['Date-Time'] > first_week_cutoff].index)
conversion_rates_2_df = conversion_rates_2_df.drop(conversion_rates_2_df[conversion_rates_2_df['Date-Time'] > first_week_cutoff].index)

model = xp.problem()

Demos = ['children', 'adults', 'retirees']

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

print('Scheduling variables added, ', time() - start_time)
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
    start[i] <= x[i][j]*(my_channel_df['Date-Time'].loc[j] - start_of_week).total_seconds()/60 + (1 - x[i][j])*(2880 - movie_db_df['runtime_with_ads'].loc[i])
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

print('Constaint 8 added, ', time() - start_time)

# # COUNTING VIEWERS GAINED ON OTHER CHANNELS
# def calculate_competitor_viewers(movie_idx, time_adv_idx, conversion_rates_i_df, movie_db_df,
#                                   channel_i_df, Demos, one_hot_genre_movie_advertized_df,
#                                     total_population=population):
#     total_viewers = 0
#     for demo in Demos:
#         # use of "DEMO_expected_view_count" instead of  "DEMO_baseline_view_count"
#         view_count = channel_i_df[f"{demo}_expected_view_count"][time_adv_idx]

#         advertized_movie_pop = movie_db_df[f"{demo.lower()}_scaled_popularity"].iloc[movie_idx]
#         demographic_popularity = channel_i_df.get(f"{demo.lower()}_popularity_factor", pd.Series([1])).iloc[time_adv_idx]
#         movie_popularity_factor = channel_i_df.get("movie_popularity_factor", pd.Series([1])).iloc[time_adv_idx]
#         # population_factor = population_factors[demo]

#         expected_viewers = (movie_popularity_factor * demographic_popularity * advertized_movie_pop *
#                        view_count * total_population)

#         # Let's compute p as a scalar product between the genres of the adv movie and those
#         #  for the actual slot
#         # p = conversion_rates_i_df.loc[time_adv_idx].drop("Date-Time").sum()
#         conversion_rates_vector = conversion_rates_i_df.loc[time_adv_idx].drop("Date-Time")
#         one_hot_vector = one_hot_genre_movie_advertized_df.loc[time_adv_idx]

#         # Compute the scalar product
#         p = np.dot(conversion_rates_vector, one_hot_vector)

#         # standard deviation based on p (total conversion), ensuring it’s zero if p >= 1 so that we choose the mean
#         standard_deviation = max(0, (1 - p) * expected_viewers)

#         # Sample viewers with precision based on p; if p >= 1, viewers = expected_viewers
#         viewers = np.random.normal(expected_viewers, standard_deviation) if standard_deviation > 0 else expected_viewers

#         # here we are trying to ensure that viewers are always less than the expected value, but not negative
#         if viewers >= expected_viewers:
#             # value chosen uniformly from the 4th quartile to ensure it's closer enough to the expected value
#             viewers = np.random.uniform(0.75 * expected_viewers, expected_viewers)

#         total_viewers += max(0, viewers) # just in case
    
#     return total_viewers

# # viewers arrays for each channel
# viewers_0 = np.zeros((number_of_movies, number_of_ad_slots_0))
# viewers_1 = np.zeros((number_of_movies, number_of_ad_slots_1))
# viewers_2 = np.zeros((number_of_movies, number_of_ad_slots_2))

# # channel 0
# for i in range(number_of_movies):
#     for r in range(number_of_ad_slots_0):
#         viewers_0[i, r] = calculate_competitor_viewers(
#             movie_idx=i,
#             time_adv_idx=r,
#             conversion_rates_i_df = conversion_rates_0_df,
#             movie_db_df=movie_db_df,
#             channel_i_df=channel_0_df,
#             Demos=Demos,
#             one_hot_genre_movie_advertized_df=one_hot_genre_movie_advertized_df,
#             total_population=population
#         )

# # channel 1
# for i in range(number_of_movies):
#     for s in range(number_of_ad_slots_1):
#         viewers_1[i, s] = calculate_competitor_viewers(
#             movie_idx=i,
#             time_adv_idx=s,
#             conversion_rates_i_df = conversion_rates_1_df,
#             movie_db_df=movie_db_df,
#             channel_i_df=channel_0_df,
#             Demos=Demos,
#             one_hot_genre_movie_advertized_df=one_hot_genre_movie_advertized_df,
#             total_population=population
#         )

# # channel 2
# for i in range(number_of_movies):
#     for t in range(number_of_ad_slots_2):
#         viewers_2[i, t] = calculate_competitor_viewers(
#             movie_idx=i,
#             time_adv_idx=t,
#             conversion_rates_i_df = conversion_rates_2_df,
#             movie_db_df=movie_db_df,
#             channel_i_df=channel_0_df,
#             Demos=Demos,
#             one_hot_genre_movie_advertized_df=one_hot_genre_movie_advertized_df,
#             total_population=population
#         )

# Precompute common factors and matrices to avoid overcalculation
def precompute_factors(movie_db_df, channel_i_df, Demos, movie_idx, time_adv_idx):
    factors = {}
    
    # Compute values that are constant across all Demos
    factors['movie_popularity_factor'] = channel_i_df.get("movie_popularity_factor", pd.Series([1])).iloc[time_adv_idx]
    factors['demographic_popularity'] = {
        demo: channel_i_df.get(f"{demo.lower()}_popularity_factor", pd.Series([1])).iloc[time_adv_idx]
        for demo in Demos
    }
    factors['advertized_movie_pop'] = {
        demo: movie_db_df[f"{demo.lower()}_scaled_popularity"].iloc[movie_idx]
        for demo in Demos
    }
    
    return factors

def calculate_competitor_viewers_optimized(movie_idx, time_adv_idx, conversion_rates_i_df, movie_db_df,
                                           channel_i_df, Demos, one_hot_genre_movie_advertized_df, total_population):
    total_viewers = 0
    # Let's precompute common factors to avoid overcalculation
    factors = precompute_factors(movie_db_df, channel_i_df, Demos, movie_idx, time_adv_idx)
    
    # vectorized operations for demographics
    conversion_rates_vector = conversion_rates_i_df.loc[time_adv_idx].drop("Date-Time")
    one_hot_vector = one_hot_genre_movie_advertized_df.loc[time_adv_idx]
    
    # Compute scalar product once
    p = np.dot(conversion_rates_vector, one_hot_vector)
    
    # Vectorization of expected view counts
    view_counts = np.array([channel_i_df[f"{demo}_expected_view_count"][time_adv_idx] for demo in Demos])
    
    for i, demo in enumerate(Demos):
        advertized_movie_pop = factors['advertized_movie_pop'][demo]
        demographic_popularity = factors['demographic_popularity'][demo]
        movie_popularity_factor = factors['movie_popularity_factor']
        
        expected_viewers = (movie_popularity_factor * demographic_popularity * advertized_movie_pop *
                            view_counts[i] * total_population)
        
        # standard_deviation = (1 - p) * expected_viewers
        # viewers = np.random.normal(expected_viewers, standard_deviation) if standard_deviation > 0 else expected_viewers

        # if viewers >= expected_viewers:
        #     viewers = np.random.uniform(0.75 * expected_viewers, expected_viewers)
        
        # total_viewers += max(0, viewers)

        # SIMPLISTIC METHOD DUE TO CONVERGENCE PROBLEMS
        total_viewers = p * expected_viewers

    
    return total_viewers

# Vectorization over all movies and ad slots
def calculate_all_viewers(number_of_movies, number_of_ad_slots, conversion_rates_i_df, movie_db_df, 
                           channel_i_df, Demos, one_hot_genre_movie_advertized_df, total_population):
    viewers = np.zeros((number_of_movies, number_of_ad_slots))
    
    for i in range(number_of_movies):
        for r in range(number_of_ad_slots):
            viewers[i, r] = calculate_competitor_viewers_optimized(
                movie_idx=i,
                time_adv_idx=r,
                conversion_rates_i_df=conversion_rates_i_df,
                movie_db_df=movie_db_df,
                channel_i_df=channel_i_df,
                Demos=Demos,
                one_hot_genre_movie_advertized_df=one_hot_genre_movie_advertized_df,
                total_population=total_population
            )
    
    return viewers

# for channel 0
viewers_0 = calculate_all_viewers(number_of_movies, number_of_ad_slots_0, conversion_rates_0_df, movie_db_df,
                                   channel_0_df, Demos, one_hot_genre_movie_advertized_df, population)
# for channel 1
viewers_1 = calculate_all_viewers(number_of_movies, number_of_ad_slots_1, conversion_rates_1_df, movie_db_df,
                                   channel_1_df, Demos, one_hot_genre_movie_advertized_df, population)
# for channel 2
viewers_2 = calculate_all_viewers(number_of_movies, number_of_ad_slots_2, conversion_rates_2_df, movie_db_df,
                                   channel_2_df, Demos, one_hot_genre_movie_advertized_df, population)


# Total viewers gained from advertising on channel 0
total_viewers_gained_0 = xp.Sum(z0[i, r] * viewers_0[i, r] for i in range(number_of_movies) for r in range(number_of_ad_slots_0))

# Total viewers gained from advertising on channel 1
total_viewers_gained_1 = xp.Sum(z1[i, s] * viewers_1[i, s] for i in range(number_of_movies) for s in range(number_of_ad_slots_1))

# Total viewers gained from advertising on channel 2
total_viewers_gained_2 = xp.Sum(z2[i, t] * viewers_2[i, t] for i in range(number_of_movies) for t in range(number_of_ad_slots_2))


print('Viewership computed after , ', time() - start_time)

total_viewers_gained = total_viewers_gained_0 + total_viewers_gained_1 + total_viewers_gained_2



# # 9. The number per thousand of viewership is less than the viewership for the time slot
# model.addConstraint(
#     u[i][j]<=
#     movie_views_for_time_slot(x, i, j, movie_db_df, my_channel_df, Demos, population)
#     for i in Movies for j in Time_slots
# )

# print('Constaint u added, ', time() - start_time)
model.addConstraint(
    uA[i] <=
    xp.Sum(
        own_advertised_views_for_time_slot(w, i , j, movie_db_df, my_channel_df, Demos, population)
        for j in Time_slots
    )
    for i in Movies
)

print('Constaint uA added, ', time() - start_time)

# model.addConstraint(
#     u0[i] <=
#     xp.Sum(
#         comp_advertised_views_for_time_slot(z0, i, r, movie_db_df, channel_0_df, Demos, population, genre_conversion_0_df)
#         for r in Ad_slots_0
#     )
#     for i in Movies
# )

# print('Constaint u0 added, ', time() - start_time)
# model.addConstraint(
#     u1[i] <=
#     xp.Sum(
#         comp_advertised_views_for_time_slot(z1, i, s, movie_db_df, channel_1_df, Demos, population, genre_conversion_1_df)
#         for s in Ad_slots_1
#     )
#     for i in Movies
# )

# print('Constaint u1 added, ', time() - start_time)
# model.addConstraint(
#     u2[i] <=
#     xp.Sum(
#         comp_advertised_views_for_time_slot(z2, i, t, movie_db_df, channel_2_df, Demos, population, genre_conversion_0_df)
#         for t in Ad_slots_2
#     )
#     for i in Movies
# )

# print('Constaint u2 added, ', time() - start_time)

# # 10. We only get contribution for viewership for movie i at time slot j if the time slot is sold
# model.addConstraint(
#     q[i][j] <= v[i][j]*(population)
#     for i in Movies for j in Time_slots
# )

# print('Constaint 10 added, ', time() - start_time)

# model.addConstraint(
#     q[i][j] == u[i][j] + uA[i] + u0[i] + u1[i] + u2[i]
#     for i in Movies for j in Time_slots
# )

TOT = total_viewers_gained + xp.Sum(uA[i] for i in Movies)



# 11. license fees and advertising slots bought must be within budget
# model.addConstraint(
#     xp.Sum(
#         y[i] * movie_db_df['license_fee'].iloc[i]
#         for i in Movies
#     )
#     + xp.Sum(
#         z0[i][r] * channel_0_df['ad_slot_price'].loc[r]
#         for i in Movies for r in Ad_slots_0
#     )
#     + xp.Sum(
#         z1[i][s] * channel_1_df['ad_slot_price'].loc[s]
#         for i in Movies for s in Ad_slots_1
#     )
#     + xp.Sum(
#         z2[i][t] * channel_2_df['ad_slot_price'].loc[t]
#         for i in Movies for t in Ad_slots_2
#     )
#     <= budget
# )

print('Constaint 11 added, ', time() - start_time)

##########################
# Objective Function
##########################

model.setObjective(
    TOT,
    sense=xp.maximize
)

print('time to intialise problem: ', time() - start_time)

# model.controls.maxtime = 300
# model.controls.maxnode = 1000
# model.controls.miprelstop = 0.01
# model.controls.tunermaxtime = 1000
# model.controls.timelimit = 30
# model.tune('g')

solvestatus, solstatus = model.optimize()

print('solve time, ', time() - start_time)

now = datetime.now()
now = str(now).replace(" ", "_")
now = now.replace(":", "-")

# saved_sol_path = f'solutions/scheduling_advert_demos_{now}'
# model.write(saved_sol_path)

x_sol = model.getSolution(x)
y_sol = model.getSolution(y)
z0_sol = model.getSolution(z0)
z1_sol = model.getSolution(z1)
z2_sol = model.getSolution(z2)
w_sol = model.getSolution(w)
v_sol = model.getSolution(v)
q_sol = model.getSolution(q)

cost = sum(y_sol[i] * movie_db_df['license_fee'].iloc[i] for i in Movies)
+ sum(z0_sol[i][r] * channel_0_df['ad_slot_price'].loc[r] for i in Movies for r in Ad_slots_0)
+ sum(z1_sol[i][s] * channel_1_df['ad_slot_price'].loc[s] for i in Movies for s in Ad_slots_1)
+ sum(z2_sol[i][t] * channel_2_df['ad_slot_price'].loc[t] for i in Movies for t in Ad_slots_2)
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
                f.write(str(q_sol[i][j]))
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