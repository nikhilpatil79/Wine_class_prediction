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

1. Install the dependencies. You can do this by running `pip install -r requirements.txt` (you'll need to create this file with the dependencies listed above).
2. Run the Streamlit app with `streamlit run main.py`.

## How it Works

The script `main.py` does the following:

1. Loads the Wine dataset from the UCI Machine Learning Repository.
2. Splits the data into training and testing sets.
3. Scales the features using StandardScaler from sklearn.
4. Trains a RandomForestClassifier on the training data.
5. The trained model can then be used to predict the class of wine.

## Future Work

- Improve the model's accuracy.
- Add more features to the Streamlit app, such as the ability to input your own data for prediction.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.