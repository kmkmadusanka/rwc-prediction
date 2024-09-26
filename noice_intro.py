import pandas as pd
import numpy as np

# Load the dataset
file_path = 'updated_autos.csv'  # Path to your dataset
data = pd.read_csv(file_path)

# Function to introduce more noise by flipping values randomly
def introduce_noise_target(column, noise_level=0.2):
    # Create a mask where noise will be introduced (based on noise level)
    mask = np.random.rand(len(data)) < noise_level
    
    # Flip 'good' to 'bad' and vice versa
    data.loc[mask, column] = data.loc[mask, column].apply(lambda x: 'bad' if x == 'good' else 'good')

# Add noise to the target columns with a higher noise level
introduce_noise_target('brake_condition', noise_level=0.3)  # 30% noise
introduce_noise_target('tire_condition', noise_level=0.3)   # 30% noise
introduce_noise_target('suspension_condition', noise_level=0.3)  # 30% noise
introduce_noise_target('emission_compliance', noise_level=0.3)  # 30% noise

# Function to introduce significant noise to numerical features
def introduce_noise_feature(column, std_dev=0.2):
    # Introduce more noise by adding random noise based on standard deviation
    noise = np.random.normal(0, std_dev, size=len(data))
    data[column] = data[column] + noise * data[column]

# Add more noise to the numerical feature columns
introduce_noise_feature('vehicle_age', std_dev=0.3)  # 30% noise to vehicle age
introduce_noise_feature('kilometer', std_dev=0.3)    # 30% noise to kilometer (mileage)

# Shuffling some rows to add randomness
data = data.sample(frac=1).reset_index(drop=True)

# Save the updated dataset with heavy noise back to CSV
noisy_file_path = 'heavily_noisy_autos.csv'
data.to_csv(noisy_file_path, index=False)

print(f"Heavily noisy dataset saved to {noisy_file_path}")
