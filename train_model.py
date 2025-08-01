import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load your dataset
df = pd.read_csv("energy_data.csv", parse_dates=[['Date', 'Time']])
df.set_index('Date_Time', inplace=True)
df['hour'] = df.index.hour

# Outlier removal
q_low = df['Global_active_power'].quantile(0.01)
q_hi = df['Global_active_power'].quantile(0.99)
df_filtered = df[(df['Global_active_power'] > q_low) & (df['Global_active_power'] < q_hi)]

# Train model
target = 'Global_active_power'
X = df_filtered.drop(target, axis=1)
y = df_filtered[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Save model
with open("energy_model.pkl", "wb") as f:
    pickle.dump(model, f)