import pandas as pd
from sklearn.model_selection import train_test_split, KFold

# Đọc dữ liệu từ file CSV
data = pd.read_csv('data/processed/grade-records_processed.csv')

# Chuẩn bị biến đầu vào (X) và đầu ra (y)
X = data[['cw1', 'mid-term', 'cw2']]  # Chọn các biến đầu vào
y = data['final']                      # Biến đầu ra

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (train/test split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gộp X_train và y_train lại thành một DataFrame để đảm bảo không mất dữ liệu
train_data = X_train.copy()
train_data['final'] = y_train

# Lưu tập train và test ra các file CSV để sử dụng sau này
train_data.to_csv('data/split/train_data.csv', index=False)
test_data = X_test.copy()
test_data['final'] = y_test
test_data.to_csv('data/split/test_data.csv', index=False)

print("Saved training and test datasets as 'train_data.csv' and 'test_data.csv'.")

# Thiết lập K-Fold trên training set
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lưu từng fold vào các tệp CSV
for fold, (train_index, val_index) in enumerate(kf.split(train_data)):
    fold_train_data = train_data.iloc[train_index].reset_index(drop=True)  # Gộp cả X và y
    fold_val_data = train_data.iloc[val_index].reset_index(drop=True)      # Gộp cả X và y

    # Lưu từng fold vào file CSV
    fold_train_data.to_csv(f'data/split/K-folds/fold_{fold}_train.csv', index=False)
    fold_val_data.to_csv(f'data/split/K-folds/fold_{fold}_val.csv', index=False)

    print(f'Saved fold {fold} train and validation data to CSV files.')
