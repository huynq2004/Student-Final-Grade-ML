import numpy as np
import ridge_model as model
import ridge_utils as utils
import ridge_plots as plot

# 1. Thiết lập các giá trị λ cần kiểm tra
lamda_values = np.logspace(-2, 1, num=100)

# 2. Huấn luyện mô hình và tìm giá trị λ tối ưu
best_lamda, avg_errors = model.train_on_folds(fold_count=5, lamda_values=lamda_values)

print(f'Giá trị λ tối ưu: {best_lamda}')

# 3. Lưu trữ mô hình với λ tốt nhất

# Sử dụng numpy để lưu trọng số vào file
X_train, y_train = utils.load_data_from('data/split/train_data.csv')
w = model.ridge_regression(X_train, y_train, best_lamda)
print('Trọng số w = ', w)
np.save('ridge_model_weights.npy', w)

# 4. Dự đoán trên tập test
# Đọc dữ liệu test
X_test, y_test = utils.load_data_from('data/split/test_data.csv')

# Dự đoán
weights = np.load('ridge_model_weights.npy')  # Tải trọng số từ file
y_pred = model.predict(X_test, weights)

# 5. Tính lỗi và vẽ biểu đồ
mse_test = np.mean((y_test - y_pred) ** 2)
print(f'Lỗi trung bình trên tập test (thủ công): {mse_test}')

# Vẽ biểu đồ lỗi theo các giá trị λ
plot.plot_errors(lamda_values, avg_errors)

# Vẽ biểu đồ so sánh điểm final dự đoán và thực tế trên tập test
plot.plot_final_scores_comparison('data/split/test_data.csv', 'ridge_model_weights.npy')
