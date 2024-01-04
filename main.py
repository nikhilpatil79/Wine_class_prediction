import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Function for model training
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# Function for model Prediction
def predict_wine_class(model, scaler, user_input):
    user_input_scaled = scaler.transform([user_input])
    prediction = model.predict(user_input_scaled)
    return prediction

# Function for displaying prediction
def display_prediction(prediction):
    wine_classes = {
        1: "Cabernet Sauvignon",
        2: "Merlot",
        3: "Chardonnay"
    }
    
    predicted_class_number = prediction[0]
    predicted_class_name = wine_classes.get(predicted_class_number, "Unknown")
    
    st.subheader(f'Predicted Wine Class: {predicted_class_number} ({predicted_class_name})')

# Streamlit app logic
def main():
    # Loading the wine dataset
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

    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='linear'),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }

    # Create a Streamlit app
    st.title('Wine Class Prediction')

    # Sidebar for model selection
    st.sidebar.title('Model Selection')
    selected_model = st.sidebar.selectbox('Select Model', list(models.keys()))

    # Selected model for predictions
    model = models[selected_model]

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Model accuracy
    st.sidebar.subheader('Model Accuracy')
    st.sidebar.write(f'Training Score: {model.score(X_train_scaled, y_train):.2f}')
    st.sidebar.write(f'Testing Score: {model.score(X_test_scaled, y_test):.2f}')

    # Display the dataset drop down and the selected dataset
    st.subheader('Wine Dataset')
    st.write(wine.data.features)

    # Input form for wine attributes
    st.header('Input Wine Attributes')

    # Define input ranges for each attribute
    alcohol = st.slider('Alcohol', min_value=11.03, max_value=14.83, value=11.03, step=0.01, format="%.2f", help="Range: 11.03 - 14.83")
    malic_acid = st.slider('Malic Acid', min_value=0.74, max_value=5.80, value=0.74, step=0.01, format="%.2f", help="Range: 0.74 - 5.80")
    ash = st.slider('Ash', min_value=1.36, max_value=3.23, value=1.36, step=0.01, format="%.2f", help="Range: 1.36 - 3.23")
    alcalinity_of_ash = st.slider('Alcalinity of Ash', min_value=10.6, max_value=30.0, value=10.6, step=0.1, format="%.1f", help="Range: 10.6 - 30.0")
    magnesium = st.slider('Magnesium', min_value=70, max_value=162, value=70, step=1, help="Range: 70 - 162")
    total_phenols = st.slider('Total Phenols', min_value=0.98, max_value=3.88, value=0.98, step=0.01, format="%.2f", help="Range: 0.98 - 3.88")
    flavanoids = st.slider('Flavanoids', min_value=0.34, max_value=5.08, value=0.34, step=0.01, format="%.2f", help="Range: 0.34 - 5.08")
    nonflavanoid_phenols = st.slider('Nonflavanoid Phenols', min_value=0.13, max_value=0.66, value=0.13, step=0.01, format="%.2f", help="Range: 0.13 - 0.66")
    proanthocyanins = st.slider('Proanthocyanins', min_value=0.41, max_value=3.58, value=0.41, step=0.01, format="%.2f", help="Range: 0.41 - 3.58")
    color_intensity = st.slider('Color Intensity', min_value=1.28, max_value=13.0, value=1.28, step=0.01, format="%.2f", help="Range: 1.28 - 13.0")
    hue = st.slider('Hue', min_value=0.48, max_value=1.71, value=0.48, step=0.01, format="%.2f", help="Range: 0.48 - 1.71")
    diluted_wines = st.slider('OD280/OD315 of Diluted Wines', min_value=1.27, max_value=4.00, value=1.27, step=0.01, format="%.2f", help="Range: 1.27 - 4.00")
    proline = st.slider('Proline', min_value=278, max_value=1680, value=278, step=1, help="Range: 278 - 1680")

    # Predict buttons 
    if st.button('Predict', key='predict_button', help="Predict using slider values"):
        user_input = [
            [alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols,
            flavanoids, nonflavanoid_phenols, proanthocyanins, color_intensity,
            hue, diluted_wines, proline]
        ]
        user_input_scaled = scaler.transform(user_input)
        prediction = model.predict(user_input_scaled)
        display_prediction(prediction)

    st.markdown('---')

    # Text area for pasting all attribute values
    input_text = st.text_area("Wine Attribute Values paste all the wine attributes separated by commas (e.g., 12.5, 3.4, 2.6, ...)")


    if st.button('Predict (Using comma separated values)', key='predict_csv_button', help="Predict using comma separated values"):
        user_input = list(map(float, input_text.split(',')))
        if len(user_input) != 13:
            st.warning("Please enter values for all 13 attributes.")
        else:
            try: 
                user_input_scaled = scaler.transform([user_input])
                prediction = model.predict(user_input_scaled)
                display_prediction(prediction)
            except ValueError:
                st.warning("Please enter valid numerical values for all 13 attributes.")

    st.markdown('---')

    # Visualization options
    visualization_options = {
        "Correlation Matrix": "correlation_matrix",
        "Alcohol vs Color Intensity": "alcohol_vs_color_intensity",
        "Accuracy vs n_estimators": "accuracy_vs_n_estimators",
        "Confusion Matrix": "confusion_matrix"
    }

    selected_option = st.selectbox("Select Visualization", list(visualization_options.keys()))

    if wine and wine.data:
        if selected_option == "Correlation Matrix":
            st.title("Correlation Matrix")
            corr = wine.data.features.corr()
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        elif selected_option == "Alcohol vs Color Intensity":
            st.title("Alcohol vs Color Intensity")
            col1, col2 = st.columns(2)

            if "Color_intensity" in wine.data.features.columns:
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(10, 10))
                    sns.scatterplot(x="Alcohol", y="Color_intensity", hue=y_np, data=wine.data.features, palette="Set1", ax=ax1)
                    st.pyplot(fig1)

                with col2:
                    fig2 = plt.figure(figsize=(10, 10))
                    ax2 = fig2.add_subplot(111, projection='3d')
                    colors = {1: 'r', 2: 'g', 3: 'b'}

                    ax2.scatter(wine.data.features["Alcohol"], wine.data.features["Color_intensity"], y_np,
                                c=[colors[x] for x in y_np])
                    ax2.set_xlabel('Alcohol')
                    ax2.set_ylabel('Color Intensity')
                    ax2.set_zlabel('Wine Class')

                    st.pyplot(fig2)
            else:
                st.write("Column 'Color_intensity' not found in the dataset.")

        elif selected_option == "Accuracy vs n_estimators":
            st.title("Accuracy vs n_estimators")
            n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
            scores = []

            for n in n_estimators:
                model = RandomForestClassifier(n_estimators=n, random_state=42)
                model.fit(X_train_scaled, y_train)
                scores.append(model.score(X_test_scaled, y_test))

            plt.figure(figsize=(25, 15))
            plt.title('Accuracy vs. Number of Trees')
            plt.xlabel('Number of Trees')
            plt.ylabel('Accuracy')
            plt.xticks(n_estimators)
            plt.plot(n_estimators, scores)
            st.pyplot(plt)

        elif selected_option == "Confusion Matrix":
            st.title("Confusion Matrices for All Models")

            # Train all models and display confusion matrices
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                st.subheader(f'Confusion Matrix for {name}')
                y_pred = model.predict(X_test_scaled)
                cm = confusion_matrix(y_test, y_pred)

                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(plt)
                st.markdown('---')

    else:
        st.write("Error: Unable to retrieve or process the dataset.")

if __name__ == "__main__":
    main()