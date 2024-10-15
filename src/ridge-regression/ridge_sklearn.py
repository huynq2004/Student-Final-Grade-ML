import numpy as np
import ridge_model as model
import ridge_utils as utils
from sklearn.linear_model import Ridge

# 1. Thiết lập các giá trị λ cần kiểm tra
lamda_values = np.logspace(-2, 1, num=100)

# 2. Huấn luyện mô hình và tìm giá trị λ tối ưu
best_lamda_sklearn, avg_errors_sklearn = model.train_on_folds_with_sklearn(fold_count=5, lamda_values=lamda_values)

print(f'Giá trị λ tốt nhất (sklearn): {best_lamda_sklearn}')


X_train, y_train = utils.load_data_from('data/split/train_data.csv')
X_test, y_test = utils.load_data_from('data/split/test_data.csv')

# 6. Sử dụng Ridge Regression từ scikit-learn
ridge_model = Ridge(alpha=best_lamda_sklearn, fit_intercept=True)
ridge_model.fit(X_train, y_train)

print("Hệ số chặn (intercept_) = " + ridge_model.intercept_)
print("Hệ số hồi quy (coef_) = " + ridge_model.coef_)

w_sklearn = np.concatenate((ridge_model.intercept_, ridge_model.coef_.flatten()))
print('Trọng số w (sklearn) = ', w_sklearn)

y_pred_sklearn = ridge_model.predict(X_test)
mse_test_sklearn = np.mean((y_test - y_pred_sklearn) ** 2)
print(f'Lỗi trung bình trên tập test (sklearn): {mse_test_sklearn}')





