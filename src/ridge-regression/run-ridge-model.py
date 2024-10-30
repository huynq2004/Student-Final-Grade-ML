import numpy as np
import pickle  # Thêm thư viện pickle
import ridge_model as model
import ridge_utils as utils
import ridge_plots as plot

# 1. Thiết lập các giá trị λ cần kiểm tra
lamda_values = np.logspace(-2, 1, num=100)

# 2. Huấn luyện mô hình và tìm giá trị λ tối ưu
best_lamda, avg_errors = model.train_on_folds(fold_count=5, lamda_values=lamda_values)

print(f'Giá trị λ tối ưu (thủ công): {best_lamda}')

# 3. Lưu trữ mô hình với λ tốt nhất
X_train, y_train = utils.load_data_from('data/split/train_data.csv')
w = model.ridge_regression(X_train, y_train, best_lamda)
print('w = ', w)

# Lưu model Ridge vào file pickle
with open('models/ridge_model.pkl', 'wb') as file:
    pickle.dump(w, file)

# 4. Dự đoán trên tập test
# Đọc dữ liệu test
X_test, y_test = utils.load_data_from('data/split/test_data.csv')

# Tải trọng số từ file pickle và dự đoán
with open('models/ridge_model.pkl', 'rb') as file:
    weights = pickle.load(file)

y_pred = model.predict(X_test, weights)

# 5. Tính lỗi và vẽ biểu đồ
mse_test = np.mean((y_test - y_pred) ** 2)
print(f'Lỗi trung bình mse trên tập test (thủ công): {mse_test}')

# Đặt ngưỡng sai số cho accuracy
error_threshold = 0.5  # Ngưỡng sai số chấp nhận được

# Tính accuracy
correct_predictions = np.sum(np.abs(y_test - y_pred) < error_threshold)
accuracy = (correct_predictions / len(y_test)) * 100

print(f'Accuracy (ngưỡng {error_threshold}): {accuracy:.2f}%')

# Vẽ biểu đồ lỗi theo các giá trị λ
plot.plot_errors(lamda_values, avg_errors)

# Vẽ biểu đồ so sánh điểm final dự đoán và thực tế trên tập test
plot.plot_final_scores_comparison(X_test, y_test, y_pred)
