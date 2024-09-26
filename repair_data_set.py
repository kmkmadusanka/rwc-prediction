import pandas as pd

# Load the dataset
file_path = 'autos.csv'  # Update with the path to your existing CSV file
data = pd.read_csv(file_path)

# Convert 'kilometer' to numeric if needed
data['kilometer'] = pd.to_numeric(data['kilometer'], errors='coerce')

# Calculate vehicle age based on 'yearOfRegistration'
current_year = 2024  # You can change this based on the actual year
data['vehicle_age'] = current_year - data['yearOfRegistration']

# 1. Simulate Brake Condition
data['brake_condition'] = data.apply(
    lambda row: 'bad' if row['kilometer'] > 150000 or row['vehicle_age'] > 15 else 'good', axis=1
)

# 2. Simulate Tire Condition
data['tire_condition'] = data.apply(
    lambda row: 'bad' if row['kilometer'] > 100000 or row['vehicle_age'] > 10 else 'good', axis=1
)

# 3. Simulate Suspension Condition
data['suspension_condition'] = data.apply(
    lambda row: 'bad' if row['notRepairedDamage'] == 'ja' or row['kilometer'] > 200000 else 'good', axis=1
)

# 4. Simulate Emission Compliance
data['emission_compliance'] = data.apply(
    lambda row: 'fail' if row['fuelType'] == 'diesel' and row['vehicle_age'] > 15 else 'pass', axis=1
)

# Save the updated dataset back to CSV
updated_file_path = 'updated_autos.csv'
data.to_csv(updated_file_path, index=False)

print(f"Updated dataset saved to {updated_file_path}")
