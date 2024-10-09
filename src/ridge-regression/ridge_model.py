import numpy as np

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
def train_on_folds(X_folds, y_folds, lamda_values):
    """
    Huấn luyện mô hình Ridge Regression trên từng fold.
    X_folds: List các tập train của từng fold.
    y_folds: List các tập nhãn train của từng fold.
    lamda_values: Danh sách các giá trị λ cần kiểm tra.
    """
    best_lamda = None
    lowest_error = float('inf')

    for lamda in lamda_values:
        fold_errors = []

        # Huấn luyện và đánh giá trên từng fold
        for i in range(len(X_folds)):
            # Chọn tập val
            X_val = X_folds[i]
            y_val = y_folds[i]

            # Chọn các fold còn lại làm train_set
            X_train = np.concatenate([X_folds[j] for j in range(len(X_folds)) if j != i])
            y_train = np.concatenate([y_folds[j] for j in range(len(y_folds)) if j != i])

            # Huấn luyện mô hình Ridge Regression
            w = ridge_regression(X_train, y_train, lamda)

            # Dự đoán trên tập val
            y_pred = np.dot(np.concatenate((np.ones((X_val.shape[0], 1)), X_val), axis=1), w)

            # Tính lỗi
            error = np.mean((y_val - y_pred) ** 2)
            fold_errors.append(error)

        # Tính lỗi trung bình qua các fold
        avg_error = np.mean(fold_errors)
        if avg_error < lowest_error:
            lowest_error = avg_error
            best_lamda = lamda

    return best_lamda

def predict(X, w):
    """
    Dự đoán đầu ra từ X với trọng số w.
    """
    ones = np.ones((X.shape[0], 1))
    Xbar = np.concatenate((ones, X), axis=1)
    return np.dot(Xbar, w)
