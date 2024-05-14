from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
import numpy as np
import joblib
import typing

TRAIN_ON_FULL = True
EXCLUDE_ROOKIES = True
EXCLUDE_EXPERT_RANKINGS = True

if TRAIN_ON_FULL:
    FULL_OR_PARTIAL = 'full'
else:
    FULL_OR_PARTIAL = 'partial'

if EXCLUDE_EXPERT_RANKINGS:
    EXPERT_POSTFIX = '_no_expert'
else:
    EXPERT_POSTFIX = ''

if EXCLUDE_ROOKIES:
    MODEL_FILE = f'models/{FULL_OR_PARTIAL}_2022_no_rookies{EXPERT_POSTFIX}.pkl'
else:
    MODEL_FILE = f'models/{FULL_OR_PARTIAL}_2022{EXPERT_POSTFIX}.pkl'

def create_dataframe(file_name: str) -> pd.DataFrame:
    return pd.read_csv(file_name)

def save_model(model, file_path):
    joblib.dump(model, file_path)


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

def train_model(X_train, y_train):
    # Create a random forest regressor
    rf = RandomForestRegressor()

    # Perform a grid search to find the best hyperparameters
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, 40, 50]
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_rf = grid_search.best_estimator_
    
    return best_rf

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

def main() -> None:
    STATS_FILE = 'stats/combined_data.csv'

    df = create_dataframe(STATS_FILE)

    player_names = df['PLAYER NAME']

    # Clean and split the data
    X, y = clean_and_split_data(df)

    if not TRAIN_ON_FULL:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = train_model(X_train, y_train)

        # Evaluate the model
        evaluate_model(model, X_test, y_test, player_names)

        # Save the model
        save_model(model, MODEL_FILE)
    else:
        # Train the model on the full dataset
        model = train_model(X, y)

        # Save the model
        save_model(model, MODEL_FILE)


if __name__ == "__main__":
    main()