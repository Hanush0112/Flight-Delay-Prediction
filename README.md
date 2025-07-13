# ✈️ Flight Delay Prediction Based on Flight and Weather Data

A machine learning-powered web application built using **Flask** that predicts whether a flight will be delayed or on time. The model considers both **flight details** and **weather conditions** (like visibility, temperature, wind, etc.) to make more accurate predictions.

---

## 🎯 Objective

To build a reliable and interactive system that predicts potential **flight delays** based on:
- Flight information (airline, departure time, route)
- Weather conditions (temperature, wind, visibility, etc.)

This helps passengers and airlines plan better and minimize disruptions.

---

## 🌟 Features

- 🛫 Predicts if a flight will be **Delayed** or **On Time**
- 🌦 Uses **real weather features** as part of input
- 🤖 Machine Learning model (Random Forest / Gradient Boosting)
- 🌐 Flask-based web interface for easy user interaction
- 📥 Takes manual input of flight and weather features
- 📊 Result shown instantly with a friendly UI

---

## 🧪 Technologies Used

| Technology                  | Purpose                                        |
|-----------------------------|------------------------------------------------|
| **Flask**                   | Backend web framework                         |
| **Pandas / NumPy**          | Data preprocessing and handling               |
| **scikit-learn**            | ML model creation, Label Encoding, Scaling    |
| **RandomForest / GBoost**   | Trained classifiers for prediction            |
| **Pickle**                  | For saving/loading trained models             |
| **HTML / Jinja**            | Templating and user form                      |


---

## 🧠 Model Input Features

- **Flight Features**
  - Airline
  - Flight Number (if used)
  - Day of Week
  - Departure Time
  - Origin Airport
  - Destination Airport

- **Weather Features**
  - Temperature
  - Visibility
  - Wind Speed
  - Weather Condition (e.g., Rain, Fog, Snow)

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Hanush0112/flight-delay-predictor.git
cd flight-delay-predictor
```
## 🧠 Model Details
Models trained using RandomForestClassifier and/or GradientBoostingClassifier

Categorical features are encoded using LabelEncoder

Data scaled using MinMaxScaler

Model evaluated with accuracy score, cross-validation, etc.

Saved using pickle and loaded into Flask backend



## 🔮 Future Enhancements
🌐 Integrate live weather APIs (e.g., OpenWeatherMap)

✈️ Integrate live flight tracking data

📱 Create a mobile version (Flutter or React Native)

💾 Save user queries and feedback

📈 Display model confidence score (probability)

☁️ Deploy on Render / Heroku / AWS / Railway
