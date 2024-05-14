from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
import numpy as np
import joblib
import typing
import pandas as pd

EXCLUDE_ROOKIES = True
EXCLUDE_EXPERT_RANKINGS = True

def clean_and_split_data(df: pd.DataFrame) -> typing.Tuple[pd.DataFrame, pd.Series]:
    # Convert the non numerical columns POS and TEAM to numerical values
    df = pd.get_dummies(df, columns=['POS', 'TEAM'])

    # For now, discard players who have NaN values
    df = df.dropna()

    if EXCLUDE_ROOKIES:
        # Remove rows where PPR_AVG_FAN PTS is 0
        df = df[df['PPR_AVG_FAN PTS'] > 0]

    if EXCLUDE_EXPERT_RANKINGS:
        # Remove any columns that contain the word RK or TIER
        df = df[df.columns.drop(list(df.filter(regex='RK|TIER')))]

    # Remove rows where the ACTUAL_PPG column is less than 3
    df = df[df['ACTUAL_PPG'] >= 3]

    # Split the data into features and target
    X = df.drop(['ACTUAL_PPG', 'PLAYER NAME'], axis=1)
    y = df['ACTUAL_PPG']

    return X, y


def evaluate_model(model, X_test, y_test, player_names):
    # sort y_test by the value in reverse order
    y_test = y_test.sort_values(ascending=False)
    
    # sort X_test by the value in reverse order
    X_test = X_test.loc[y_test.index]

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Calculate the mean absolute error
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae}')

    # Get the median error
    median_error = np.median(np.abs(y_test - y_pred))
    print(f'Median Error: {median_error}')

    # Calculate the R^2 score
    r2 = r2_score(y_test, y_pred)
    print(f'R^2 Score: {r2}')

    # Get the feature importances
    feature_importances = model.feature_importances_
    feature_importances = pd.Series(feature_importances, index=X_test.columns)
    feature_importances = feature_importances.sort_values(ascending=False)
    print(f'Feature Importances: {feature_importances}')

    # Perform cross-validation
    scores = cross_val_score(model, X_test, y_test, cv=5)
    print(f'Cross-Validation Scores: {scores}')
    print(f'Mean Cross-Validation Score: {np.mean(scores)}')

    # Print the predicted value and actual value, as well as the player name based on the index from the sorted y_test
    for i in range(len(y_pred)):
        print(f'{player_names[y_test.index[i]]}: Predicted: {y_pred[i]:.2f}, Actual: {y_test.iloc[i]:.2f}, Difference: {abs(y_pred[i] - y_test.iloc[i]):.2f}')
    
    # Do the same print but only if the difference is less than 2
    print('\nPlayers with a difference of less than 2')
    for i in range(len(y_pred)):
        if abs(y_pred[i] - y_test.iloc[i]) < 2:
            print(f'{player_names[y_test.index[i]]}: Predicted: {y_pred[i]:.2f}, Actual: {y_test.iloc[i]:.2f}, Difference: {abs(y_pred[i] - y_test.iloc[i]):.2f}')

def main():
    # load model
    model = joblib.load('models/full_2022_no_rookies_no_expert.pkl')

    # load the 2023 dataset
    df = pd.read_csv('stats/2023.csv')

    # load the truth values from 2023_final_ppr_ppg.csv
    truth = pd.read_csv('stats/2023_final_ppr_ppg.csv')

    # Rename AVG column to ACTUAL_PPG
    truth = truth.rename(columns={'AVG': 'ACTUAL_PPG'})
    # Rename Player column to PLAYER NAME
    truth = truth.rename(columns={'Player': 'PLAYER NAME'})

    # combine the AVG column in the truth dataset with the 2023 dataset keyed on name
    df = df.merge(truth, how='left', left_on='PLAYER NAME', right_on='PLAYER NAME')

    # clean and split the data
    X, y = clean_and_split_data(df)

    # drop TEAM_TEAM
    X = X.drop('TEAM_TEAM', axis=1)
    
    # Test on the full dataset
    evaluate_model(model, X, y, df['PLAYER NAME'])


if __name__ == "__main__":
    main()