import pandas as pd
import numpy as np
import random

# Load the movie dataset (replace 'movie_database.csv' with your actual file)
movie_db_df = pd.read_csv('data/filtered_movie_database_135.csv', parse_dates=['release_date'])

# Ensure the dataset contains a 'runtime_with_ads' column in minutes
movie_db_df['runtime_with_ads'] = movie_db_df['runtime_with_ads'].astype(int)

# Constants
DAILY_LIMIT = 1020  # Daily time limit in minutes (17 hours)
WEEKS = 12          # Duration of the schedule
DAYS_PER_WEEK = 7   # Number of days in a week
TOTAL_DAYS = WEEKS * DAYS_PER_WEEK

# Function to simulate one day's schedule
def simulate_day_schedule(movie_db):
    remaining_time = DAILY_LIMIT
    day_schedule = []
    
    while remaining_time > 0:
        # Randomly pick a movie
        movie = movie_db.sample(1).iloc[0]
        runtime = movie['runtime_with_ads']
        
        # Check if the movie fits within the remaining time
        if runtime <= remaining_time:
            day_schedule.append(movie['title'])
            remaining_time -= runtime
        else:
            break
    
    return day_schedule, DAILY_LIMIT - remaining_time

# Monte Carlo Simulation for multiple days
def monte_carlo_simulation(movie_db, total_days):
    all_schedules = []
    total_utilization = []
    
    for _ in range(total_days):
        day_schedule, utilized_time = simulate_day_schedule(movie_db)
        all_schedules.append(day_schedule)
        total_utilization.append(utilized_time / DAILY_LIMIT)  # Utilization as percentage
    
    return all_schedules, total_utilization

# Run the simulation
schedules, utilization = monte_carlo_simulation(movie_db_df, TOTAL_DAYS)

# Output results
for day, schedule in enumerate(schedules, start=1):
    print(f"Day {day}: {', '.join(schedule)}")
    
average_utilization = np.mean(utilization) * 100
print(f"\nAverage daily utilization: {average_utilization:.2f}%")