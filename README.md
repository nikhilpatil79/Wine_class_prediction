# Wine Class Prediction

This project uses machine learning to predict the class of wine based on various features. The model is trained on the UCI Machine Learning Repository's Wine dataset.

## Dependencies

- streamlit
- pandas
- sklearn
- ucimlrepo
- matplotlib
- seaborn

## How to Run

1. Install the dependencies.
2. Run the Streamlit app with `streamlit run main.py`.

## How it Works

The script `main.py` does the following:

1. Loads the Wine dataset from the UCI Machine Learning Repository.
2. Splits the data into training and testing sets.
3. Scales the features using StandardScaler from sklearn.
4. Trains a RandomForestClassifier on the training data.
5. The trained model can then be used to predict the class of wine.

## Live Demo
You can try out a live demo of the project at [https://wineclassprediction-saikapil-mini-project.streamlit.app/](https://wineclassprediction-saikapil-mini-project.streamlit.app/).

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.