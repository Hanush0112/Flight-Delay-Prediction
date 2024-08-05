from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and label encoders
try:
    model = pickle.load(open('model.pkl', 'rb'))
    le_dict = pickle.load(open('label_encoders.pkl', 'rb'))
    le_carrier = le_dict['carrier']
    le_dest = le_dict['dest']
    le_origin = le_dict['origin']
except Exception as e:
    print(f"Error loading model or label encoders: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        year = request.form['year']
        month = request.form['month']
        day = request.form['day']
        carrier = request.form['carrier']
        origin = request.form['origin']
        dest = request.form['dest']
        
        # Convert inputs to appropriate types
        year = int(year)
        month = int(month)
        day = int(day)
        carrier = str(carrier)
        origin = str(origin)
        dest = str(dest)

        if year >= 2013:
            x1 = [year, month, day, carrier, origin, dest]
            df1 = pd.DataFrame(data=[x1], columns=['year', 'month', 'day', 'carrier', 'origin', 'dest'])
            
            df1['carrier'] = le_carrier.transform(df1['carrier'])
            df1['origin'] = le_origin.transform(df1['origin'])
            df1['dest'] = le_dest.transform(df1['dest'])

            x = df1.iloc[:, :6].values
            ans = model.predict(x)
            output = ans[0]
        else:
            output = "Year should be 2013 or later."
        
        return render_template('index.html', prediction_text=f'Prediction: {"Delayed" if output == 1 else "Not Delayed"}')
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    try:
        app.run(debug=False)
    except KeyboardInterrupt:
        print("Application interrupted. Exiting gracefully...")
