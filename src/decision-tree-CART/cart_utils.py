import numpy as np
import pandas as pd

# Hàm đọc dữ liệu từ file CSV
def load_fold_data(fold_number):
    """Đọc dữ liệu của một fold từ file CSV."""
    train_data = pd.read_csv(f'data/split/K-folds2/fold_{fold_number}_train.csv')
    val_data = pd.read_csv(f'data/split/K-folds2/fold_{fold_number}_val.csv')

    # Tách biến đầu vào (cw1, mid-term, cw2) và biến đầu ra (final)
    X_train = train_data[['cw1', 'mid-term', 'cw2']].values
    y_train = train_data['final'].values

    X_val = val_data[['cw1', 'mid-term', 'cw2']].values
    y_val = val_data['final'].values

    return X_train, y_train, X_val, y_val

def load_data_from(path):
    input = pd.read_csv(f'{path}')
    data = input[['cw1', 'mid-term', 'cw2']].values
    label = input[['final']].values
    return data, label
