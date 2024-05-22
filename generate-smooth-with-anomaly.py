import pandas as pd
import numpy as np

def generate_smooth_dataset(start_timestamp, end_timestamp):
    # Create date range
    date_range = pd.date_range(start=start_timestamp, end=end_timestamp, freq='S')

    # Generate random walk for each parameter
    num_seconds = len(date_range)
    print(num_seconds)
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

    # Introduce anomalies
    anomaly_indices = np.random.choice(num_seconds, size=int(num_seconds * 0.05), replace=False)  # 5% of data as anomalies
    print(anomaly_indices)
    for idx in anomaly_indices:
        # Randomly select a parameter to perturb
        param = 'heart_rate'

        # Perturb the parameter value
        print('before', df.loc[idx, param])
        df.loc[idx, param] += np.random.normal(0, 30)  # Adjust the standard deviation to control the severity of anomalies
        print('after', df.loc[idx, param])

    return df


# Example usage
start_timestamp = pd.Timestamp('2024-05-01')
end_timestamp = pd.Timestamp('2024-05-01 00:15:00')
dataset = generate_smooth_dataset(start_timestamp, end_timestamp)

# Save to CSV
dataset.to_csv('smooth_dataset_with_anomalies_after19.csv', index=False)
