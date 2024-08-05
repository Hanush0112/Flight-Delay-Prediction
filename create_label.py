import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load dataset
df = pd.read_csv('Data/Processed_data15.csv')

# Initialize and fit label encoders
le_carrier = LabelEncoder()
df['carrier'] = le_carrier.fit_transform(df['carrier'])

le_dest = LabelEncoder()
df['dest'] = le_dest.fit_transform(df['dest'])

le_origin = LabelEncoder()
df['origin'] = le_origin.fit_transform(df['origin'])

# Save label encoders
with open('label_encoders.pkl', 'wb') as le_file:
    pickle.dump({
        'carrier': le_carrier,
        'dest': le_dest,
        'origin': le_origin
    }, le_file)

print("Label encoders saved successfully.")
