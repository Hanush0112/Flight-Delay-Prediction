import pickle
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('Data/Processed_data15.csv')

# Load model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load label encoders
try:
    le_dict = pickle.load(open('label_encoders.pkl', 'rb'))
    le_carrier = le_dict['carrier']
    le_dest = le_dict['dest']
    le_origin = le_dict['origin']
except Exception as e:
    print(f"Error loading label encoders: {e}")
    exit()

# Apply label encoders to the dataset
df['carrier'] = le_carrier.transform(df['carrier'])
df['dest'] = le_dest.transform(df['dest'])
df['origin'] = le_origin.transform(df['origin'])

# Prepare features and target variable
X = df.iloc[:, 0:6].values
y = df['delayed']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=61)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
