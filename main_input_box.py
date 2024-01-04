import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ucimlrepo import fetch_ucirepo 


# Loding the wine dataset
wine = fetch_ucirepo(id=109) 
  
# data (as pandas dataframes) 
X = wine.data.features 
y = wine.data.targets


# Convert y to a NumPy array before using train_test_split
y_np = y.values.ravel() if isinstance(y, pd.DataFrame) else y.ravel()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_np, test_size=0.2, random_state=42)

# Scale the features if necessary
scaler = StandardScaler(with_mean=True, with_std=True)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)


# Create a Streamlit app
st.title('Wine Class Prediction')

#classes that the model is trained to predict
# display one after the other with images
st.header('Wine Classes')
st.subheader('Class 1: Cabernet Sauvignon')
st.subheader('Class 2: Merlot')
st.subheader('Class 3: Chardonnay')


# Input form for wine attributes
st.header('Input Wine Attributes')


# Define input ranges for each attribute
alcohol = st.number_input('Alcohol', min_value=11.03, max_value=14.83, step=0.01, value=11.03, format="%.2f", help="Range: 11.03 - 14.83")
malic_acid = st.number_input('Malic Acid', min_value=0.74, max_value=5.80, step=0.01, value=0.74, format="%.2f", help="Range: 0.74 - 5.80")
ash = st.number_input('Ash', min_value=1.36, max_value=3.23, step=0.01, value=1.36, format="%.2f", help="Range: 1.36 - 3.23")
alcalinity_of_ash = st.number_input('Alcalinity of Ash', min_value=10.6, max_value=30.0, step=0.1, value=10.6, format="%.1f", help="Range: 10.6 - 30.0")
magnesium = st.number_input('Magnesium', min_value=70, max_value=162, step=1, value=70, help="Range: 70 - 162")
total_phenols = st.number_input('Total Phenols', min_value=0.98, max_value=3.88, step=0.01, value=0.98, format="%.2f", help="Range: 0.98 - 3.88")
flavanoids = st.number_input('Flavanoids', min_value=0.34, max_value=5.08, step=0.01, value=0.34, format="%.2f", help="Range: 0.34 - 5.08")
nonflavanoid_phenols = st.number_input('Nonflavanoid Phenols', min_value=0.13, max_value=0.66, step=0.01, value=0.13, format="%.2f", help="Range: 0.13 - 0.66")
proanthocyanins = st.number_input('Proanthocyanins', min_value=0.41, max_value=3.58, step=0.01, value=0.41, format="%.2f", help="Range: 0.41 - 3.58")
color_intensity = st.number_input('Color Intensity', min_value=1.28, max_value=13.0, step=0.01, value=1.28, format="%.2f", help="Range: 1.28 - 13.0")
hue = st.number_input('Hue', min_value=0.48, max_value=1.71, step=0.01, value=0.48, format="%.2f", help="Range: 0.48 - 1.71")
diluted_wines = st.number_input('OD280/OD315 of Diluted Wines', min_value=1.27, max_value=4.00, step=0.01, value=1.27, format="%.2f", help="Range: 1.27 - 4.00")
proline = st.number_input('Proline', min_value=278, max_value=1680, step=1, value=278, help="Range: 278 - 1680")

# Predict button
if st.button('Predict'):
    user_input = [[alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols,
                   flavanoids, nonflavanoid_phenols, proanthocyanins, color_intensity,
                   hue, diluted_wines, proline]]
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)
    
    wine_classes = {
        1: "Cabernet Sauvignon",
        2: "Merlot",
        3: "Chardonnay"
    }
    
    predicted_class_number = prediction[0]
    predicted_class_name = wine_classes.get(predicted_class_number, "Unknown")
    
    st.subheader(f'Predicted Wine Class: {predicted_class_number} ({predicted_class_name})')










