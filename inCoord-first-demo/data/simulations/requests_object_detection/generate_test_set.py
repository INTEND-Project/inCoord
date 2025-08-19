import pandas as pd
import numpy as np
import csv
from datetime import datetime, timedelta
import random

MEAN_VALUE = 50

def calculate_actual_value(deviation_percentage, mean):
    """Convert percentage deviation to actual value"""
    return mean * (1 + deviation_percentage / 100)


def interpolate(start_value, end_value, fraction):
    """Linear interpolation between two values"""
    return start_value + (end_value - start_value) * fraction


def generate_time_series_data():
    # Define approximate deviations from the heatmap (in percentage)
    # Rows represent hours (0-23), columns represent days (Monday-Sunday)
    hourly_deviation_matrix = [
        # Mon   Tue    Wed    Thu    Fri    Sat    Sun
        [40, 20, 60, 30, 10, 5, 40],  # Hour 0
        [30, 10, 40, 20, 5, 0, 30],  # Hour 1
        [0, -10, 10, 0, -10, -15, 0],  # Hour 2
        [-20, -30, -20, -20, -30, -35, -20],  # Hour 3
        [-40, -50, -40, -40, -50, -55, -40],  # Hour 4
        [-70, -80, -70, -70, -80, -80, -70],  # Hour 5
        [-90, -95, -90, -90, -95, -90, -90],  # Hour 6
        [-70, -80, -70, -70, -80, -75, -70],  # Hour 7
        [-40, -50, -40, -40, -50, -45, -40],  # Hour 8
        [-20, -30, -20, -20, -30, -25, -20],  # Hour 9
        [0, -10, 0, 0, -10, -5, 0],  # Hour 10
        [0, -5, 0, 0, -5, 0, 0],  # Hour 11
        [5, 0, 5, 5, 0, 5, 5],  # Hour 12
        [10, 5, 10, 10, 5, 10, 10],  # Hour 13
        [30, 20, 30, 30, 20, 30, 30],  # Hour 14
        [60, 50, 60, 60, 50, 60, 60],  # Hour 15
        [80, 70, 80, 80, 70, 80, 80],  # Hour 16
        [60, 50, 60, 60, 50, 60, 60],  # Hour 17
        [50, 40, 40, 50, 40, 50, 50],  # Hour 18
        [40, 30, 30, 40, 30, 40, 40],  # Hour 19
        [30, 20, 20, 30, 20, 30, 30],  # Hour 20
        [20, 10, 10, 20, 10, -20, 20],  # Hour 21
        [30, 20, 20, 30, 10, -5, 30],  # Hour 22
        [50, 40, 45, 50, 30, 10, 50]  # Hour 23
    ]

    # Days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Create the time series data with 5-minute intervals
    time_series_data = []
    mean = MEAN_VALUE  # The specified mean value
    intervals_per_hour = 12  # 12 five-minute intervals per hour

    # Generate data for one week with 5-minute intervals
    for day_index, day in enumerate(days_of_week):
        for hour in range(24):
            # Get the current hour's deviation percentage
            current_hour_deviation = hourly_deviation_matrix[hour][day_index]

            # Get the next hour's deviation percentage (wrapping to the next day if needed)
            next_hour = (hour + 1) % 24
            next_day_index = (day_index + 1) % 7 if next_hour == 0 else day_index
            next_hour_deviation = hourly_deviation_matrix[next_hour][next_day_index]

            # Create entries for each 5-minute interval within this hour
            for interval in range(intervals_per_hour):
                # Calculate the fraction of the hour completed
                fraction = interval / intervals_per_hour

                # Interpolate the deviation percentage for this interval
                interpolated_deviation = interpolate(current_hour_deviation, next_hour_deviation, fraction)

                # Calculate the actual value
                actual_value = calculate_actual_value(interpolated_deviation, mean)

                # Add some minor random variation to make the data more realistic
                # Using a small variance (0.5) to keep it close to the interpolated value
                random_variation = (random.random() - 0.5) * 0.5

                # Round to one decimal place for more precision
                request_count = int(round((actual_value + random_variation) * 10) / 10)

                # Calculate minutes
                minutes = interval * 5

                # Format time as HH:MM
                time_string = f"{hour:02d}:{minutes:02d}"

                # Add entry to the database
                time_series_data.append({
                    'day': day,
                    'time': time_string,
                    'request_count': request_count,
                    'deviation_percentage': round(interpolated_deviation * 10) / 10
                })

    return time_series_data


def create_csv_file(data, filename='user_requests_timeseries.csv'):
    """Create a CSV file from the time series data"""
    # Create a pandas DataFrame
    df = pd.DataFrame(data)

    # Rename columns to match the desired format
    df = df.rename(columns={
        'day': 'Day',
        'time': 'Time',
        'request_count': 'RequestCount',
        'deviation_percentage': 'DeviationPercentage'
    })

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"CSV file has been saved as {filename}")

    # Calculate and print statistics
    print(f"Total data points: {len(df)}")
    print(f"Mean request count: {df['RequestCount'].mean():.2f}")
    print(f"Min request count: {df['RequestCount'].min()}")
    print(f"Max request count: {df['RequestCount'].max()}")

    return df


def main():
    # Generate time series data
    time_series_data = generate_time_series_data()

    # Create CSV file
    df = create_csv_file(time_series_data)

    # Display the first 10 rows
    print("\nFirst 10 rows of data:")
    print(df.head(10))


if __name__ == "__main__":
    main()