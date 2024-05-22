import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

MEDIAN_HEART_RATE = 75
MEDIAN_SYSTOLIC_BLOOD_PRESSURE = 120
MEDIAN_DIASTOLIC_BLOOD_PRESSURE = 75
MEDIAN_TEMPERATURE = 36.5
MEDIAN_OXYGEN_SATURATION = 98
MEDIAN_RESPIRATORY_RATE = 16
MEDIAN_GLUCOSE = 85

MEDIAN_HEART_RATE_STANDARD_DEVIATION = 0.3
MEDIAN_SYSTOLIC_BLOOD_PRESSURE_STANDARD_DEVIATION = 0.1
MEDIAN_DIASTOLIC_BLOOD_PRESSURE_STANDARD_DEVIATION = 0.05
MEDIAN_TEMPERATURE_STANDARD_DEVIATION = 0.02
MEDIAN_OXYGEN_SATURATION_STANDARD_DEVIATION = 0.02
MEDIAN_RESPIRATORY_RATE_STANDARD_DEVIATION = 0.075
MEDIAN_GLUCOSE_STANDARD_DEVIATION = 0.15


def introduce_anomaly(df, anomaly_labels_df, anomaly_index, anomaly_duration, param, std_adjustment):
    for i in range(anomaly_duration):
        if anomaly_index + i < len(df):
            df.loc[anomaly_index + i, param] += std_adjustment
            anomaly_labels_df.loc[anomaly_index + i, param] = 1

    return df


def introduce_fever_anomaly(df, anomaly_labels_df, anomaly_index, anomaly_duration):
    # Perturb the parameter value
    adjusted_heart_rate_std = random.uniform(50, 80)
    # Adjusting systolic and diastolic blood pressure standard deviations based on heart rate standard deviation
    adjusted_systolic_std = MEDIAN_SYSTOLIC_BLOOD_PRESSURE_STANDARD_DEVIATION * (
            adjusted_heart_rate_std / MEDIAN_HEART_RATE_STANDARD_DEVIATION)
    adjusted_diastolic_std = MEDIAN_DIASTOLIC_BLOOD_PRESSURE_STANDARD_DEVIATION * (
            adjusted_heart_rate_std / MEDIAN_HEART_RATE_STANDARD_DEVIATION)
    adjusted_temperature_std = random.uniform(1.5, 2.5)
    adjusted_respiratory_rate_std = random.uniform(3, 8)

    print('adjusted_heart_rate_std', adjusted_heart_rate_std)
    print('adjusted_systolic_std', adjusted_systolic_std)
    print('adjusted_diastolic_std', adjusted_diastolic_std)
    print('adjusted_temperature_std', adjusted_temperature_std)
    print('adjusted_respiratory_rate_std', adjusted_respiratory_rate_std)

    # Perturb the parameter values for the anomaly duration
    df = introduce_anomaly(df, anomaly_labels_df, anomaly_index, anomaly_duration, 'heart_rate',
                           adjusted_heart_rate_std)
    df = introduce_anomaly(df, anomaly_labels_df, anomaly_index, anomaly_duration, 'systolic_blood_pressure',
                           adjusted_systolic_std)
    df = introduce_anomaly(df, anomaly_labels_df, anomaly_index, anomaly_duration, 'diastolic_blood_pressure',
                           adjusted_diastolic_std)
    df = introduce_anomaly(df, anomaly_labels_df, anomaly_index, anomaly_duration, 'temperature',
                           adjusted_temperature_std)
    df = introduce_anomaly(df, anomaly_labels_df, anomaly_index, anomaly_duration, 'respiratory_rate',
                           adjusted_respiratory_rate_std)

    return df


def introduce_hypoglycemia_anomaly(df, anomaly_labels_df, anomaly_index, anomaly_duration):
    # Perturb the parameter value
    adjusted_glucose_std = np.random.normal(-20, 10)
    # Adjusting heart rate and respiratory rate standard deviations based on glucose standard deviation
    adjusted_heart_rate_std = random.uniform(40, 60)
    adjusted_respiratory_rate_std = np.random.normal(10, 2)

    print('adjusted_glucose_std', adjusted_glucose_std)
    print('adjusted_heart_rate_std', adjusted_heart_rate_std)
    print('adjusted_respiratory_rate_std', adjusted_respiratory_rate_std)

    # Perturb the parameter values for the anomaly duration
    df = introduce_anomaly(df, anomaly_labels_df, anomaly_index, anomaly_duration, 'glucose', adjusted_glucose_std)
    df = introduce_anomaly(df, anomaly_labels_df, anomaly_index, anomaly_duration, 'heart_rate',
                           adjusted_heart_rate_std)
    df = introduce_anomaly(df, anomaly_labels_df, anomaly_index, anomaly_duration, 'respiratory_rate',
                           adjusted_respiratory_rate_std)

    return df


def introduce_asthma_anomaly(df, anomaly_labels_df, anomaly_index, anomaly_duration):
    # Perturb the parameter value
    adjusted_respiratory_rate_std = np.random.normal(20, 10)
    adjusted_oxygen_saturation_std = np.random.normal(-10, 5)

    print('adjusted_respiratory_rate_std', adjusted_respiratory_rate_std)
    print('adjusted_oxygen_saturation_std', adjusted_oxygen_saturation_std)

    # Perturb the parameter values for the anomaly duration
    df = introduce_anomaly(df, anomaly_labels_df, anomaly_index, anomaly_duration, 'respiratory_rate',
                           adjusted_respiratory_rate_std)
    df = introduce_anomaly(df, anomaly_labels_df, anomaly_index, anomaly_duration, 'oxygen_saturation',
                           adjusted_oxygen_saturation_std)

    return df


def generate_smooth_dataset(start_timestamp, end_timestamp):
    np.random.seed(random.randint(0, 100))  # For reproducibility
    # Create date range
    date_range = pd.date_range(start=start_timestamp, end=end_timestamp, freq='s')

    # Generate random walk for each parameter
    num_seconds = len(date_range)
    print('num_seconds', num_seconds)

    heart_rate = np.cumsum(np.random.normal(0, MEDIAN_HEART_RATE_STANDARD_DEVIATION, num_seconds)) + MEDIAN_HEART_RATE
    heart_rate = np.clip(heart_rate, 60, 100)

    systolic_blood_pressure = np.cumsum(np.random.normal(0, MEDIAN_SYSTOLIC_BLOOD_PRESSURE_STANDARD_DEVIATION,
                                                         num_seconds)) + MEDIAN_SYSTOLIC_BLOOD_PRESSURE
    systolic_blood_pressure = np.clip(systolic_blood_pressure, 105, 140)

    diastolic_blood_pressure = np.cumsum(np.random.normal(0, MEDIAN_DIASTOLIC_BLOOD_PRESSURE_STANDARD_DEVIATION,
                                                          num_seconds)) + MEDIAN_DIASTOLIC_BLOOD_PRESSURE
    diastolic_blood_pressure = np.clip(diastolic_blood_pressure, 65, 85)

    temperature = np.cumsum(
        np.random.normal(0, MEDIAN_TEMPERATURE_STANDARD_DEVIATION, num_seconds)) + MEDIAN_TEMPERATURE
    temperature = np.clip(temperature, 35.5, 37.5)

    oxygen_saturation = np.cumsum(
        np.random.normal(0, MEDIAN_OXYGEN_SATURATION_STANDARD_DEVIATION, num_seconds)) + MEDIAN_OXYGEN_SATURATION
    oxygen_saturation = np.clip(oxygen_saturation, 94, 100)

    respiratory_rate = np.cumsum(
        np.random.normal(0, MEDIAN_RESPIRATORY_RATE_STANDARD_DEVIATION, num_seconds)) + MEDIAN_RESPIRATORY_RATE
    respiratory_rate = np.clip(respiratory_rate, 12, 20)

    glucose = np.cumsum(np.random.normal(0, MEDIAN_GLUCOSE_STANDARD_DEVIATION, num_seconds)) + MEDIAN_GLUCOSE
    glucose = np.clip(glucose, 70, 100)

    # Create DataFrame
    df = pd.DataFrame({
        'id': np.arange(num_seconds),
        'timestamp': date_range,
        'heart_rate': heart_rate,
        'systolic_blood_pressure': systolic_blood_pressure,
        'diastolic_blood_pressure': diastolic_blood_pressure,
        'temperature': temperature,
        'oxygen_saturation': oxygen_saturation,
        'respiratory_rate': respiratory_rate,
        'glucose': glucose
    })

    anomaly_labels_df = pd.DataFrame({
        'timestamp': date_range,
        'heart_rate': np.zeros(num_seconds),
        'systolic_blood_pressure': np.zeros(num_seconds),
        'diastolic_blood_pressure': np.zeros(num_seconds),
        'temperature': np.zeros(num_seconds),
        'oxygen_saturation': np.zeros(num_seconds),
        'respiratory_rate': np.zeros(num_seconds),
        'glucose': np.zeros(num_seconds)
    })

    # Introduce fever
    fever_anomaly_index = np.random.choice(num_seconds // 10)  # Random index for the anomaly
    print('anomaly_index', fever_anomaly_index)
    fever_anomaly_duration = np.random.randint(90, 150)
    print('anomaly_duration', fever_anomaly_duration)
    df = introduce_fever_anomaly(df, anomaly_labels_df, fever_anomaly_index, fever_anomaly_duration)
    #
    # Introduce hypoglycemia
    hypoglycemia_anomaly_index = np.random.choice(
        np.arange(fever_anomaly_index + fever_anomaly_duration + 30, num_seconds // 2))  # Random index for the anomaly
    print('anomaly_index', hypoglycemia_anomaly_index)
    hypoglycemia_anomaly_duration = np.random.randint(90, 150)
    print('anomaly_duration', hypoglycemia_anomaly_duration)
    df = introduce_hypoglycemia_anomaly(df, anomaly_labels_df, hypoglycemia_anomaly_index, hypoglycemia_anomaly_duration)

    # Introduce asthma
    asthma_anomaly_index = np.random.choice(np.arange(hypoglycemia_anomaly_index + hypoglycemia_anomaly_duration + 30,
                                                      3 * num_seconds // 2))  # Random index for the anomaly
    print('anomaly_index', asthma_anomaly_index)
    asthma_anomaly_duration = np.random.randint(90, 150)
    print('anomaly_duration', asthma_anomaly_duration)
    df = introduce_asthma_anomaly(df, anomaly_labels_df, asthma_anomaly_index, asthma_anomaly_duration)

    # Introduce fever
    fever_anomaly_index = np.random.choice(
        np.arange(asthma_anomaly_index + asthma_anomaly_duration + 90, num_seconds))  # Random index for the anomaly
    print('anomaly_index', fever_anomaly_index)
    fever_anomaly_duration = np.random.randint(90, 150)
    print('anomaly_duration', fever_anomaly_duration)
    df = introduce_fever_anomaly(df, anomaly_labels_df, fever_anomaly_index, fever_anomaly_duration)
    #
    # Smooth out the data again
    df['heart_rate'] = df['heart_rate'].rolling(window=15, min_periods=1).mean()
    anomaly_labels_df['heart_rate'] = anomaly_labels_df['heart_rate'].rolling(window=15, min_periods=1).mean().round(0)

    df['systolic_blood_pressure'] = df['systolic_blood_pressure'].rolling(window=15, min_periods=1).mean()
    anomaly_labels_df['systolic_blood_pressure'] = anomaly_labels_df['systolic_blood_pressure'].rolling(window=15,
                                                                                                        min_periods=1).mean().round(0)

    df['diastolic_blood_pressure'] = df['diastolic_blood_pressure'].rolling(window=15, min_periods=1).mean()
    anomaly_labels_df['diastolic_blood_pressure'] = anomaly_labels_df['diastolic_blood_pressure'].rolling(window=15,
                                                                                                          min_periods=1).mean().round(0)

    df['temperature'] = df['temperature'].rolling(window=15, min_periods=1).mean()
    anomaly_labels_df['temperature'] = anomaly_labels_df['temperature'].rolling(window=15, min_periods=1).mean().round(0)

    df['oxygen_saturation'] = df['oxygen_saturation'].rolling(window=15, min_periods=1).mean()
    anomaly_labels_df['oxygen_saturation'] = anomaly_labels_df['oxygen_saturation'].rolling(window=15,
                                                                                            min_periods=1).mean().round(0)

    df['respiratory_rate'] = df['respiratory_rate'].rolling(window=15, min_periods=1).mean()
    anomaly_labels_df['respiratory_rate'] = anomaly_labels_df['respiratory_rate'].rolling(window=15,
                                                                                          min_periods=1).mean().round(0)

    df['glucose'] = df['glucose'].rolling(window=15, min_periods=1).mean()
    anomaly_labels_df['glucose'] = anomaly_labels_df['glucose'].rolling(window=15, min_periods=1).max()

    return np.round(df, 2), anomaly_labels_df


# Example usage
start_timestamp = pd.Timestamp('2024-05-01')
end_timestamp = pd.Timestamp('2024-05-01 02:00:00')
# end_timestamp = pd.Timestamp('2024-05-01 01:06:30')
dataset, anomaly_labels = generate_smooth_dataset(start_timestamp, end_timestamp)

# Save to CSV
dataset.to_csv('new_dataset_with_labels.csv', index=False)
anomaly_labels.to_csv('anomaly_labels.csv', index=False)

# fever: heart rate raise, blood pressure raise, temperature raise, high respiratory rate
# hypoglycemia: low glucose, high heart rate, high respiratory rate
# asthma: high respiratory rate, low oxygen saturation

# panic attack: high heart rate, high respiratory rate, high blood pressure
