# run_ridge_model.py
import numpy as np
import ridge_model as model
import ridge_utils as utils
import ridge_plots as plot

# 1. Thiết lập các giá trị λ cần kiểm tra
lamda_values = np.logspace(-2, 1, num=100)

# 2. Huấn luyện mô hình và tìm giá trị λ tối ưu
best_lamda, avg_errors = model.train_on_folds(fold_count=5, lamda_values=lamda_values)#lưu 2 cái gt này

print(f'GT lamda best: {best_lamda}')

# 3. Lưu trữ mô hình với λ tốt nhất

# Sử dụng numpy để lưu trọng số vào file
X_train, y_train = utils.load_data_from('D:/Project_ML/Student-Final-Grade-ML/data/split/train_data.csv')
w = model.ridge_regression(X_train, y_train, best_lamda)
print('w = ', w)
np.save('ridge_model_weights.npy', w)

# 4. Dự đoán trên tập test
# Đọc dữ liệu test
X_test, y_test = utils.load_data_from('D:/Project_ML/Student-Final-Grade-ML/data/split/test_data.csv')
# Dự đoán
weights = np.load('ridge_model_weights.npy')  # Tải trọng số từ file, gọi hàm thôi
y_pred = model.predict(X_test, weights)

# 5. Tính lỗi và vẽ biểu đồ
mse_test = np.mean((y_test - y_pred) ** 2)
print(f'Error mean test: {mse_test}')

# Vẽ biểu đồ lỗi theo các giá trị λ
plot.plot_errors(lamda_values, avg_errors)

# Vẽ biểu đồ so sánh điểm final dự đoán và thực tế trên tập test
plot.plot_final_scores_comparison(X_test, y_test,weights)