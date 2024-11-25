import pandas as pd
import numpy as np
import xpress as xp
from datetime import datetime, timedelta
from time import time
import os

from time_slot_viewership import movie_views_for_time_slot,comp_advertised_views_for_time_slot, own_advertised_views_for_time_slot, calculate_ad_slot_price


n_competitors = 3
# weekly data loading
def load_weekly_data(week_number):
    base_path = "data/"
    week_prefix = f"WEEK_{week_number}_"
    
    my_channel_df = pd.read_csv(f'{base_path}{week_prefix}channel_A_schedule.csv', parse_dates=['Date-Time'])
    movie_db_df = pd.read_csv(f'{base_path}movie_database_with_license_fee_100.csv', parse_dates=['release_date'])
    movie_genre_df = pd.read_csv(f'{base_path}movie_genre_hot_one_100.csv')

    channel_dfs = {
        i: pd.read_csv(f'{base_path}ADVERTS_{week_prefix}channel_{i}_schedule.csv', parse_dates=['Date-Time'])
        for i in range(n_competitors)
    }

    conversion_rate_dfs = {
        i: pd.read_csv(f'{base_path}{week_prefix}channel_{i}_conversion_rates.csv', parse_dates=['Date-Time'])
        for i in range(n_competitors)
    }
    
    return my_channel_df, movie_db_df, movie_genre_df, channel_dfs, conversion_rate_dfs

# load previous week's results
def load_previous_results(week_number):
    if week_number == 1:
        return None

    previous_week = week_number - 1
    result_path = f"./output/week_{previous_week}/viewership.csv"
    
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Previous week results not found for Week {previous_week}. Please run it first.")
    
    return pd.read_csv(result_path)

# save weekly results (TODO : BUT WE CAN LOOK AT WHAT WE DID AT THE END OF THIS)
def save_weekly_results(model, week_number, my_channel_df, movie_db_df, viewers_gained, x, z0, z1, z2, channel_dfs):
    output_dir = f"./output/week_{week_number}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save scheduling decisions (TODO)
    with open(f"{output_dir}/schedule_results.txt", "w") as f:
        # movies
        f.write("Movie Schedule:\n")
        for j in range(len(my_channel_df)):
            for i in range(len(movie_db_df)):
                if model.getSolution(x[i][j]) == 1:
                    f.write(f"At {my_channel_df['Date-Time'].iloc[j]} show movie {movie_db_df['title'].iloc[i]}\n")
        
        # advertisements
        f.write("\nAdvertisements:\n")
        for channel_id, z in zip(range(n_competitors), [z0, z1, z2]):
            f.write(f"Channel {channel_id} Advertisements:\n")
            ad_slots = range(len(channel_dfs[channel_id]))
            for i in range(len(movie_db_df)):
                for r in ad_slots:
                    if model.getSolution(z[i][r]) == 1:
                        f.write(f"At {channel_dfs[channel_id]['Date-Time'].iloc[r]} advertise movie {movie_db_df['title'].iloc[i]} on Channel {channel_id}\n")
    
    # Save viewership gained (TODO????)
    pd.DataFrame([{"Week": week_number, "Viewers Gained": viewers_gained}]).to_csv(f"{output_dir}/viewership.csv", index=False)

# start and end of the week
def get_week_cutoff_dates(start_date, week_number):
    # Our week 1 starts at this date : 2024-10-01
    week_start = start_date + timedelta(weeks=(week_number - 1))
    week_end = week_start + timedelta(days=6, hours=23, minutes=59) # We need to look at this carefully
    return week_start, week_end

# single week scheduling
def schedule_week(week_number):
    start_time = time()
    print(f"--- Scheduling for Week {week_number} ---")

    # Useful to keep track of the time taken by different approaches
    now_start_time = datetime.now()
    now_start_time = str(now_start_time).replace(" ", "_")
    now_start_time = now_start_time.replace(":", "-")

    # Load data for the current week
    my_channel_df, movie_db_df, movie_genre_df, channel_dfs, conversion_rate_dfs = load_weekly_data(week_number)
    
    # load previous week's results if not the first week
    if week_number > 1:
        previous_results = load_previous_results(week_number)
        # this is to modify the baseline viewership using the previous week's results (TODO)
        for i in range(len(my_channel_df)):
            my_channel_df.loc[i, "Children Baseline View Count"] += previous_results["Viewers Gained"].iloc[0] * 0.1

    # dynamic week cutoff dates
    start_date = datetime(2024, 10, 1)  # starting date of Week 1
    week_start, week_end = get_week_cutoff_dates(start_date, week_number)

    # filter data for the current week
    my_channel_df = my_channel_df[(my_channel_df['Date-Time'] >= week_start) & (my_channel_df['Date-Time'] <= week_end)]

    model = xp.problem()

    Demos = ['children', 'adults', 'retirees']
    Genres = movie_genre_df.columns.to_list()

    number_of_movies = len(movie_db_df.index)
    number_of_time_slots = len(my_channel_df.index)
    number_of_comp_channels = 3
    number_of_ad_slots_0 = len(conversion_rate_dfs[0])
    number_of_ad_slots_1 = len(conversion_rate_dfs[1])
    number_of_ad_slots_2 = len(conversion_rate_dfs[2])

    Movies = range(number_of_movies)
    Time_slots = range(number_of_time_slots)
    Channels = range(number_of_comp_channels)

    Ad_slots_0 = range(number_of_ad_slots_0)
    Ad_slots_1 = range(number_of_ad_slots_1)
    Ad_slots_2 = range(number_of_ad_slots_2)

    channel_dict = {
        0: channel_dfs[0],
        1: channel_dfs[1],
        2: channel_dfs[2]
    }

    conversion_dict = {
        0: conversion_rate_dfs[0],
        1: conversion_rate_dfs[1],
        2: conversion_rate_dfs[2]
    }

    population = 1000000
    viewership_units = 1000 # IF THIS IS EVOLVING LINEARLY, WE CAN EXPRESS IT PER UNIT INSTEAD OF 1000
    ad_sell_price_per_unit = 100
    budget = 1000000

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

    # whether movie i is advertised on channel 0 at time slot j
    z0 = np.array(
        [xp.var(name=f'z0_{i}_{r}', vartype=xp.binary) for i in Movies for r in Ad_slots_0],
        dtype=xp.npvar
    ).reshape(number_of_movies,number_of_ad_slots_0)
    model.addVariable(z0)

    # whether movie i is advertised on channel 1 at time slot j
    z1 = np.array(
        [xp.var(name=f'z1_{i}_{s}', vartype=xp.binary) for i in Movies for s in Ad_slots_1],
        dtype=xp.npvar
    ).reshape(number_of_movies,number_of_ad_slots_1)
    model.addVariable(z1)

    # whether movie i is advertises on channel 2 at time slot j
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
    # v = np.array(
    #     [xp.var(name=f"v_{i}_{j}", vartype=xp.binary) for i in Movies for j in Time_slots]
    # ).reshape(number_of_movies,number_of_time_slots)
    # model.addVariable(v)

    # sum of viewers for movie i across time slots shown 
    u = [xp.var(name=f"u_{i}", vartype=xp.integer) for i in Movies] # THIS NEEDS TO BE CAREFULLY DEFINED
    model.addVariable(u)

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

    # THE BIG-M (2880 BELOW) NEEDS TO BE WELL-DEFINED (LIKE A PARAMETER) 
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

    # model.addConstraint(
    #     xp.Sum(w[i][j] for i in Movies) + xp.Sum(v[i][j] for i in Movies) == 1 for j in Time_slots
    # )


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
    # LOGICALLY WE NEED ALSO TO ADD THIS (THE CONDITION HOLDS FOR OUR OWN CHANNEL TOO)

    # model.addConstraint(
    #     w[i][j] <= y[i] for i in Movies for j in Time_slots
    # )

    # # 7. Only sell ad slot for movie i at time j if it is shown then
    # model.addConstraint(
    #     v[i][j] <= x[i][j] for i in Movies for j in Time_slots
    # )

    # 8. Only advertise before the movie is scheduled
    model.addConstraint(
        z0[i][r]*(conversion_rate_dfs[0]['Date-Time'].loc[r] - start_of_week + timedelta(minutes=30)).total_seconds()/60 <= start[i]
        for i in Movies for r in Ad_slots_0
    )
    model.addConstraint(
        z1[i][s]*(conversion_rate_dfs[1]['Date-Time'].loc[s] - start_of_week + timedelta(minutes=30)).total_seconds()/60 <= start[i]
        for i in Movies for s in Ad_slots_1
    )
    model.addConstraint(
        z2[i][t]*(conversion_rate_dfs[2]['Date-Time'].loc[t] - start_of_week + timedelta(minutes=30)).total_seconds()/60 <= start[i]
        for i in Movies for t in Ad_slots_2
    )
    # ON OUR OWN CHANNEL TOO.....

    # model.addConstraint(
    #     w[i][j]*(my_channel_df['Date-Time'].loc[j] - start_of_week + timedelta(minutes=30)).total_seconds()/60 <= start[i]
    #     for i in Movies for j in Time_slots
    # )

    print('Advertising constraints added, ', time() - start_time)

    # WE CAN JUST COUNT PER UNIT, IT DOESN'T MATTER DUE TO LINEARITY
    # # 9. The number per thousand of viewership is less than the viewership for the time slot
    # model.addConstraint(
    #     u[i]*viewership_units <=
    #     xp.Sum(
    #         movie_views_for_time_slot(x, i, j, movie_db_df, my_channel_df, Demos, population)
    #         for j in Time_slots
    #     )
    #     # + xp.Sum(
    #     #     own_advertised_views_for_time_slot(w, i , j, movie_db_df, my_channel_df, Demos, population)
    #     #     for j in Time_slots
    #     # )
    #     + xp.Sum(
    #         comp_advertised_views_for_time_slot(z0, i, r, movie_db_df, channel_dfs[0], conversion_rate_dfs[0], Demos, Genres, population)
    #         for r in Ad_slots_0
    #     )
    #     + xp.Sum(
    #         comp_advertised_views_for_time_slot(z1, i, s, movie_db_df, channel_dfs[0], conversion_rate_dfs[0], Demos, Genres, population)
    #         for s in Ad_slots_1
    #     )
    #     + 
    #     xp.Sum(
    #         comp_advertised_views_for_time_slot(z2, i, t, movie_db_df, channel_dfs[0], conversion_rate_dfs[0], Demos, Genres, population)
    #         for t in Ad_slots_2
    #     )
    #     for i in Movies
    # )

    # # Function to select a random true count column for each demographic
    # def random_view_count(channel_i_df, demographic):
    #     """ demographic will come from "Demos" list 
    #         and the counts from 'channel_dfs[0]' which are the competitor's advert slots databases
    #     """
    #     # retirees_true_view_count_1
    #     true_counts = [f"{demographic}_true_view_count_{i}" for i in range(0, 10)]
    #     selected_column = np.random.choice(true_counts)
    #     return channel_i_df[selected_column]

    # Our assumption about demographic population factors (distribution)
    population_factors = {
        "children": 0.3,
        "adults": 0.5,
        "retirees": 0.2
    }

    def calculate_competitor_viewers(movie_idx, time_adv_idx, conversion_rates_i_df, movie_db_df, channel_i_df,
                                    Demos, population_factors, total_population=population):
        total_viewers = 0
        for demo in Demos:
            # # select randomly view count for the demographic
            # view_count = random_view_count(channel_i_df, demo)[time_adv_idx]

            # use of "DEMO_baseline_view_count" instead of selecting randomly view true view count for the demographic
            view_count = channel_i_df[f"{demo}_baseline_view_count"][time_adv_idx]

            advertized_movie_pop = movie_db_df[f"{demo.lower()}_scaled_popularity"].iloc[movie_idx]
            demographic_popularity = channel_i_df.get(f"{demo.lower()}_popularity_factor", pd.Series([1])).iloc[time_adv_idx]
            movie_popularity_factor = channel_i_df.get("movie_popularity_factor", pd.Series([1])).iloc[time_adv_idx]
            population_factor = population_factors[demo]

            expected_viewers = (movie_popularity_factor * demographic_popularity * advertized_movie_pop *
                        view_count * population_factor * total_population)

            # Let's compute p as the sum of conversion rates for the given time slot
            p = conversion_rates_i_df.loc[time_adv_idx].drop("Date-Time").sum()

            # standard deviation based on p (total conversion), ensuring it’s zero if p >= 1 so that we choose the mean
            standard_deviation = max(0, (1 - p) * expected_viewers)

            # Sample viewers with precision based on p; if p >= 1, viewers = expected_viewers
            viewers = np.random.normal(expected_viewers, standard_deviation) if standard_deviation > 0 else expected_viewers

            # here we are trying to ensure that viewers are always less than the expected value, but not negative
            if viewers >= expected_viewers:
                # value chosen uniformly from the 4th quartile to ensure it's closer enough to the expected value
                viewers = np.random.uniform(0.75 * expected_viewers, expected_viewers)

            total_viewers += max(0, viewers) # just in case
        
        return total_viewers

    # viewers arrays for each channel
    viewers_0 = np.zeros((number_of_movies, number_of_ad_slots_0))
    viewers_1 = np.zeros((number_of_movies, number_of_ad_slots_1))
    viewers_2 = np.zeros((number_of_movies, number_of_ad_slots_2))

    # channel 0
    for i in range(number_of_movies):
        for r in range(number_of_ad_slots_0):
            viewers_0[i, r] = calculate_competitor_viewers(
                movie_idx=i,
                time_adv_idx=r,
                conversion_rates_i_df = conversion_rate_dfs[0],
                movie_db_df=movie_db_df,
                channel_i_df=channel_dfs[0],
                Demos=Demos,
                population_factors=population_factors,
                total_population=population
            )

    # channel 1
    for i in range(number_of_movies):
        for s in range(number_of_ad_slots_1):
            viewers_1[i, s] = calculate_competitor_viewers(
                movie_idx=i,
                time_adv_idx=s,
                conversion_rates_i_df = conversion_rate_dfs[1],
                movie_db_df=movie_db_df,
                channel_i_df=channel_dfs[0],
                Demos=Demos,
                population_factors=population_factors,
                total_population=population
            )

    # channel 2
    for i in range(number_of_movies):
        for t in range(number_of_ad_slots_2):
            viewers_2[i, t] = calculate_competitor_viewers(
                movie_idx=i,
                time_adv_idx=t,
                conversion_rates_i_df = conversion_rate_dfs[2],
                movie_db_df=movie_db_df,
                channel_i_df=channel_dfs[0],
                Demos=Demos,
                population_factors=population_factors,
                total_population=population
            )

    # Total viewers gained from advertising on channel 0
    total_viewers_gained_0 = xp.Sum(z0[i, r] * viewers_0[i, r] for i in range(number_of_movies) for r in range(number_of_ad_slots_0))

    # Total viewers gained from advertising on channel 1
    total_viewers_gained_1 = xp.Sum(z1[i, s] * viewers_1[i, s] for i in range(number_of_movies) for s in range(number_of_ad_slots_1))

    # Total viewers gained from advertising on channel 2
    total_viewers_gained_2 = xp.Sum(z2[i, t] * viewers_2[i, t] for i in range(number_of_movies) for t in range(number_of_ad_slots_2))



    # # 12. We only get contribution for viewership for movie i at time slot j if the time slot is sold
    # model.addConstraint(
    #     u[i] <= v[i][j]*(population/viewership_units)
    #     for i in Movies for j in Time_slots
    # )


    print('Viewership computed after , ', time() - start_time)

    # 12. license fees and advertising slots bought must be within budget
    # model.addConstraint(
    #     xp.Sum(
    #         y[i] * movie_db_df['license_fee'].iloc[i]
    #         for i in Movies
    #     )
    #     + xp.Sum(
    #         z[i][j][c] * calculate_ad_slot_price(j, channel_dict[c])
    #         for i in Movies for j in Time_slots for c in Channels
    #     )
    #     <= budget
    # )


    ##########################
    # Objective Function
    ##########################

    # model.setObjective(
    #     xp.Sum(ad_sell_price_per_unit*u[i]  for i in Movies),
    #     sense=xp.maximize
    # )

    total_viewers_gained = total_viewers_gained_0 + total_viewers_gained_1 + total_viewers_gained_2
    objective_function = total_viewers_gained # - total_ad_cost
    model.setObjective(objective_function, sense=xp.maximize)

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
    + sum(model.getSolution(z0[i][j]) * calculate_ad_slot_price(j, channel_dfs[0]) for i in Movies for j in Ad_slots_0)
    + sum(model.getSolution(z1[i][j]) * calculate_ad_slot_price(j, channel_dfs[1]) for i in Movies for j in Ad_slots_1)
    + sum(model.getSolution(z2[i][j]) * calculate_ad_slot_price(j, channel_dfs[2]) for i in Movies for j in Ad_slots_2)
    print(cost)
    # if solstatus != xp.SolStatus.INFEASIBLE or solstatus != xp.SolStatus.UNBOUNDED or solstatus != xp.SolStatus.UNBOUNDED:

    output_dir = "./output"
    # create the dir if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Solve the model
    solvestatus, solstatus = model.optimize()
    if solstatus != xp.SolStatus.OPTIMAL:
        print(f"Optimization did not converge for Week {week_number}. Status: {solstatus}")
        return

    # Get weekly viewership
    weekly_viewers = model.getObjVal() # WE SHOULD DIVIDE THIS BY THE AVERAGE PROFIT PER VIEWER GAINED

    # Save results
    save_weekly_results(model, week_number, my_channel_df, movie_db_df, weekly_viewers, z0, z1, z2, channel_dfs)
    print(f"Week {week_number} completed. Viewers gained: {weekly_viewers}")
    
    now = datetime.now()
    now = str(now).replace(" ", "_")
    now = now.replace(":", "-")
    
    with open(f"./output/output_proba1_{str(now)}_WEEK_{week_number}.txt", "w") as f:
        f.write(f'From {now_start_time} to {now}')
        f.write('\n')
        f.write('VIEWERSHIP: ')
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
        # for j in Time_slots:
        #     for i in Movies:
        #         if model.getSolution(w[i][j]) == 1:
        #             f.write("At ")
        #             f.write(str(my_channel_df['Date-Time'].loc[j]))
        #             f.write(" on own channel advertise movie ")
        #             f.write(movie_db_df['title'].loc[i])
        #             f.write('\n')
        f.write('AD SLOTS: ')
        f.write('\n')

        for j in Ad_slots_0:
            for i in Movies:
                if model.getSolution(z0[i][j]) == 1:
                    f.write("At ")
                    f.write(str(channel_dfs[0]['Date-Time'].loc[j]))
                    f.write(" on channel 0 advertise movie ")
                    f.write(movie_db_df['title'].loc[i])
                    f.write('\n')
        for j in Ad_slots_1:
            for i in Movies:
                if model.getSolution(z1[i][j]) == 1:
                    f.write("At ")
                    f.write(str(channel_dfs[1]['Date-Time'].loc[j]))
                    f.write(" on channel 1 advertise movie ")
                    f.write(movie_db_df['title'].loc[i])
                    f.write('\n')
        for j in Ad_slots_2:
            for i in Movies:
                if model.getSolution(z2[i][j]) == 1:
                    f.write("At ")
                    f.write(str(channel_dfs[2]['Date-Time'].loc[j]))
                    f.write(" on channel 2 advertise movie ")
                    f.write(movie_db_df['title'].loc[i])
                    f.write('\n')
    f.close()
    print("PROCESS SUCCESSFULLY COMPLETED")

# EXAMPLE
week_to_run = 1
schedule_week(week_to_run)
