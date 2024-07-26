import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def make_windows(df, target_col, train_end, test_start, output_dim):
    
    #df['Lag_'+str(output_dim)] = df[target_col].shift(output_dim)
    df.dropna(inplace=True)

    target = df[[target_col]]
    df.drop(target_col, axis=1, inplace=True)

    rawX_train = df[:train_end].values
    rawX_test = df[test_start:].values

    rawY_train = target[:train_end]
    rawY_test = target[test_start:]

    #scaler_x = StandardScaler()
    X_train = rawX_train#scaler_x.fit_transform(rawX_train)
    X_test = rawX_test#scaler_x.transform(rawX_test)

    #scaler_y = StandardScaler()
    Y_train = rawY_train #scaler_y.fit_transform(rawY_train)
    Y_test = rawY_test #scaler_y.transform(rawY_test)

    X_train = np.array([X_train[i:i + output_dim] for i in range(X_train.shape[0]-output_dim+1)])
    X_test = np.array([X_test[i:i + output_dim] for i in range(X_test.shape[0]-output_dim+1)])

    y_train = np.array([Y_train[i:i + output_dim] for i in range(Y_train.shape[0]-output_dim+1)]).reshape(-1, output_dim)
    y_test = np.array([Y_test[i:i + output_dim] for i in range(Y_test.shape[0]-output_dim+1)]).reshape(-1, output_dim)
    
    return X_train, y_train, X_test, y_test # shape = (len(x), output_dim, n_features)


def make_flat_windows(df, target_col, train_end, test_start, output_dim):
    
    #df['Lag_'+str(output_dim)] = df[target_col].shift(output_dim)
    df.dropna(inplace=True)

    target = df[[target_col]]
    df.drop(target_col, axis=1, inplace=True)
    
    feature_dim = len(df.columns)

    rawX_train = df[:train_end].values
    rawX_test = df[test_start:].values

    rawY_train = target[:train_end]
    rawY_test = target[test_start:]

    #scaler_x = StandardScaler()
    X_train = rawX_train#scaler_x.fit_transform(rawX_train)
    X_test = rawX_test#scaler_x.transform(rawX_test)

    #scaler_y = StandardScaler()
    Y_train = rawY_train #scaler_y.fit_transform(rawY_train)
    Y_test = rawY_test #scaler_y.transform(rawY_test)

    X_train = np.array([X_train[i:i + output_dim] for i in range(X_train.shape[0]-output_dim+1)]).reshape(-1, output_dim * feature_dim)
    X_test = np.array([X_test[i:i + output_dim] for i in range(X_test.shape[0]-output_dim+1)]).reshape(-1, output_dim * feature_dim)

    y_train = np.array([Y_train[i:i + output_dim] for i in range(Y_train.shape[0]-output_dim+1)]).reshape(-1, output_dim)
    y_test = np.array([Y_test[i:i + output_dim] for i in range(Y_test.shape[0]-output_dim+1)]).reshape(-1, output_dim)
    
    return X_train, y_train, X_test, y_test # shape = (len(x), output_dim, n_features)


def germansolarfarm(df, datetime_col, target_col):
    
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df.set_index(datetime_col, inplace=True)
    
    return df


def europewindfarm(df, datetime_col, target_col):
    
    df.drop('ForecastingTime', axis=1, inplace=True)
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df.set_index(datetime_col, inplace=True)
    
    return df