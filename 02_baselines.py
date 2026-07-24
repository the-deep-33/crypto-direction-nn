import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from loader import load_dataset
import os

FEATURES = ['r_lag_1', 'r_lag_2', 'r_lag_3', 'r_lag_4', 'r_lag_5']


def walk_forward(df, model, feature_cols, initial_train=500, step=50):
    X = df[feature_cols].values
    y = df['label'].values
    n = len(df)

    all_preds = []
    all_truth = []

    boundary = initial_train
    while boundary < n:
        test_end = min(boundary + step, n)

        X_train, y_train = X[:boundary], y[:boundary]
        X_test,  y_test  = X[boundary:test_end], y[boundary:test_end]
        n_test = test_end - boundary

        if model == 'majority':
            print(f"boundary={boundary}, ups={sum(y_train):.0f}, half={boundary/2:.0f}, "
                f"predict={'1' if sum(y_train) > boundary/2 else '0'}")
            # predict the training-set majority class (past only), constant over the block
            if sum(y_train) > boundary / 2:
                preds = n_test * [1]
            else:
                preds = n_test * [0]
        else:
            # scale using TRAIN statistics only, then apply to test (no leakage)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)

        all_preds.extend(preds)
        all_truth.extend(y_test)

        boundary = test_end

    return np.array(all_preds), np.array(all_truth)


if __name__ == "__main__":

    print("-------------------------------------------")
    print("Valid arguments: ")
    print("1: Looking at crypto records day by day")
    print("2: Looking at crypto records hour by hour")
    print("-------------------------------------------")
    print("Your choice: ")

    choice = input()

    if choice == "1":
        timeframe = "1d"
    elif choice == "2":
        timeframe = "1h"
    else:
        raise ValueError("Invalid argument!")

    df = load_dataset(timeframe)

    print(len(df))
    print(df['date'].iloc[0], '->', df['date'].iloc[-1])



    df['log_return'] = np.log(df['close']).diff()

    future_ret = df['log_return'].shift(-1)
    df['label'] = (future_ret > 0).astype(float)
    df.loc[future_ret.isna(), 'label'] = np.nan

    for k in range(1, 6):
        df[f'r_lag_{k}'] = df['log_return'].shift(k)

    df = df.dropna().reset_index(drop=True)

    models = {
        'majority':  'majority',
        'logistic':  LogisticRegression(max_iter=1000),
        'gbt':       GradientBoostingClassifier(),
    }

    print(f"\n{'model':<12}{'n_preds':<10}{'accuracy':<10}")
    for name, mdl in models.items():
        preds, truth = walk_forward(df, mdl, FEATURES)
        acc = (preds == truth).mean()
        print(f"{name:<12}{len(preds):<10}{acc:.4f}")