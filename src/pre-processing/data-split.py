import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


# Đọc dữ liệu từ file CSV
data = pd.read_csv('Student-Final-Grade-ML/data/processed/grade-records_processed.csv')

# Chuẩn bị biến đầu vào (X) và đầu ra (y)
X = data[['cw1', 'mid-term', 'cw2']]  # Chọn các biến đầu vào
y = data['final']                      # Biến đầu ra

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# In ra kích thước của các tập dữ liệu
print(f'Size of training set: {X_train.shape[0]}')
print(f'Size of test set: {X_test.shape[0]}')

# Kiểm tra các phần tử đầu tiên của tập huấn luyện và kiểm tra
print("Training set features:")
print(X_train.head())

print("\nTraining set target:")
print(y_train.head())

print("\nTest set features:")
print(X_test.head())

print("\nTest set target:")
print(y_test.head())

# Lưu tập huấn luyện
train_data = pd.DataFrame(X_train)
train_data['final'] = y_train  # Thêm biến đầu ra vào tập huấn luyện
train_data.to_csv('Student-Final-Grade-ML/data/split/train_data.csv', index=False)

# Lưu tập kiểm tra
test_data = pd.DataFrame(X_test)
test_data['final'] = y_test  # Thêm biến đầu ra vào tập kiểm tra
test_data.to_csv('Student-Final-Grade-ML/data/split/test_data.csv', index=False)

print("Train and test datasets saved as 'train_data.csv' and 'test_data.csv'.")


# Thiết lập K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lưu từng fold vào các tệp CSV
for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
    X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

    # Tạo DataFrame cho tập huấn luyện và tập kiểm tra của fold hiện tại
    fold_train_data = pd.DataFrame(X_fold_train)
    fold_train_data['final'] = y_fold_train.reset_index(drop=True)  # Thêm biến đầu ra

    fold_val_data = pd.DataFrame(X_fold_val)
    fold_val_data['final'] = y_fold_val.reset_index(drop=True)  # Thêm biến đầu ra

    # Lưu vào tệp CSV
    fold_train_data.to_csv(f'Student-Final-Grade-ML/data/split/K-folds/fold_{fold}_train.csv', index=False)
    fold_val_data.to_csv(f'Student-Final-Grade-ML/data/split/K-folds/fold_{fold}_val.csv', index=False)

    print(f'Saved fold {fold} train and validation data to CSV files.')
