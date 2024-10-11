import numpy as np
import pandas as pd
import ridge_utils as utils

# Hàm thuật toán
def ridge_regression(X, y, lamda):
    """
    X: Ma trận thuộc tính  (các cột cw1, mid-term, cw2).
    y: nhãn (cột final).
    lamda: Tham số regularization λ.
    """
    # Thêm cột bias (cột toàn số 1)
    ones = np.ones((X.shape[0], 1))
    Xbar = np.concatenate((ones, X), axis=1)

    # Tính trọng số Ridge Regression theo công thức
    A = np.dot(Xbar.T, Xbar) + lamda * np.identity(Xbar.shape[1])
    b = np.dot(Xbar.T, y)
    w = np.dot(np.linalg.pinv(A), b)

    return w

# Hàm huấn luyện
# Hàm huấn luyện Ridge Regression trên từng fold
def train_on_folds(fold_count, lamda_values):
    """
    Huấn luyện mô hình Ridge Regression trên từng fold từ các file CSV.
    fold_count: Số lượng fold (5 folds).
    lamda_values: Danh sách các giá trị lamda cần kiểm tra.
    """
    best_lamda = None
    lowest_error = float('inf')
    avg_errors = []  # Danh sách lưu trữ lỗi trung bình cho mỗi λ
    
    for lamda in lamda_values:
        fold_errors = []

        # Huấn luyện và đánh giá trên từng fold
        for i in range(fold_count):
            # Đọc dữ liệu cho fold thứ i
            X_train, y_train, X_val, y_val = utils.load_fold_data(i)

            # Huấn luyện mô hình Ridge Regression
            w = ridge_regression(X_train, y_train, lamda)

            # Dự đoán trên tập val
            y_pred = np.dot(np.concatenate((np.ones((X_val.shape[0], 1)), X_val), axis=1), w)

            # Tính lỗi (Mean Squared Error)
            error = np.mean((y_val - y_pred) ** 2)
            fold_errors.append(error)


        # Tính lỗi trung bình qua các fold
        avg_error = np.mean(fold_errors)
        avg_errors.append(avg_error)  # Lưu lỗi trung bình vào danh sách
        
        if avg_error < lowest_error:
            lowest_error = avg_error
            best_lamda = lamda

        print(f"Lambda = {lamda}, Average Fold Error = {avg_error}")

    return best_lamda, avg_errors

def predict(X, w):
    """
    Dự đoán đầu ra từ X với trọng số w.
    """
    ones = np.ones((X.shape[0], 1))
    Xbar = np.concatenate((ones, X), axis=1)
    return np.dot(Xbar, w)
