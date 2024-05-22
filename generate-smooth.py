import pandas as pd
import numpy as np


def generate_smooth_dataset(start_date, end_date):
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='S')

    # Generate random walk for each parameter
    num_seconds = len(date_range)
    heart_rate = np.cumsum(np.random.normal(0, 0.3, num_seconds)) + 75
    systolic_blood_pressure = np.cumsum(np.random.normal(0, 0.1, num_seconds)) + 120
    diastolic_blood_pressure = np.cumsum(np.random.normal(0, 0.05, num_seconds)) + 80
    temperature = np.cumsum(np.random.normal(0, 0.05, num_seconds)) + 98.6
    oxygen_saturation = np.cumsum(np.random.normal(0, 0.02, num_seconds)) + 98
    respiratory_rate = np.cumsum(np.random.normal(0, 0.1, num_seconds)) + 16

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': date_range,
        'heart_rate': heart_rate,
        'systolic_blood_pressure': systolic_blood_pressure,
        'diastolic_blood_pressure': diastolic_blood_pressure,
        'temperature': temperature,
        'oxygen_saturation': oxygen_saturation,
        'respiratory_rate': respiratory_rate
    })

    return df


# Example usage
start_timestamp = pd.Timestamp('2024-05-01')
end_timestamp = pd.Timestamp('2024-05-01 00:05:00')
dataset = generate_smooth_dataset(start_timestamp, end_timestamp)

# Save to CSV
dataset.to_csv('smooth_dataset.csv', index=False)

# TODO change all random std to use random.uniform
