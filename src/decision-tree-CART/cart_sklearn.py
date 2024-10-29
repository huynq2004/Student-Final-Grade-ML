import numpy as np
from sklearn.tree import DecisionTreeRegressor
import cart_utils as utils
import cart_plots as plot

# 1. Thiết lập độ sâu cây cần kiểm tra và ngưỡng dừng
depth_values = range(1, 11)
threshold = 0.01  # Ngưỡng chênh lệch lỗi để dừng


# 2. Huấn luyện mô hình và tìm độ sâu tối ưu
errors = []
best_depth = None
lowest_error = float('inf')
previous_error = None  # Lưu lỗi của độ sâu trước đó

for depth in depth_values:
    fold_errors = []
    for fold in range(5):
        X_train, y_train, X_val, y_val = utils.load_fold_data(fold)
        # Khởi tạo và huấn luyện Decision Tree bằng thư viện scikit-learn
        tree = DecisionTreeRegressor(max_depth=depth)
        tree.fit(X_train, y_train)
        # Dự đoán trên tập val
        y_pred = tree.predict(X_val)
        # Tính lỗi MSE
        mse = np.mean((y_val - y_pred) ** 2)
        fold_errors.append(mse)
    avg_error = np.mean(fold_errors)
    errors.append(avg_error)
    #Kiểm tra ngưỡng dừng
    if previous_error is not None and abs(previous_error - avg_error) < threshold:
        print(f"Chênh lệch lỗi nhỏ hơn ngưỡng {threshold}, dừng tại độ sâu {depth}.")
        break
    # Cập nhật độ sâu tối ưu nếu lỗi thấp hơn
    if avg_error < lowest_error:
        lowest_error = avg_error
        best_depth = depth
    previous_error = avg_error  # Cập nhật lỗi trước đó để kiểm tra ngưỡng
    print(f"Depth = {depth}, Average Fold Error = {avg_error}")

# 3. Lưu trữ mô hình với độ sâu tốt nhất
print(f'Độ sâu tối ưu (sklearn): {best_depth}')


# 4. Dự đoán trên tập test
X_train, y_train = utils.load_data_from('C:/Users/ADMIN/Documents/HocMay/Student-Final-Grade-ML/data/split/train_data.csv')
X_test, y_test = utils.load_data_from('C:/Users/ADMIN/Documents/HocMay/Student-Final-Grade-ML/data/split/test_data.csv')
tree = DecisionTreeRegressor(max_depth=best_depth)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

# 5. Tính lỗi và vẽ biểu đồ
mse_test = np.mean((y_test - y_pred) ** 2)
print(f'Lỗi trung bình trên tập test (sklearn): {mse_test}')

# Đặt ngưỡng sai số cho accuracy
error_threshold = 0.5  # Ngưỡng sai số chấp nhận được

# Tính accuracy
correct_predictions = np.sum(np.abs(y_test - y_pred) < error_threshold)
accuracy = (correct_predictions / len(y_test) ) 

print(f'Accuracy (ngưỡng {error_threshold}): {accuracy:.2f}%')

# Vẽ biểu đồ lỗi theo độ sâu
plot.plot_errors(depth_values[:len(errors)], errors)

# Vẽ cây kết quả
plot.plot_decision_tree(tree, feature_names=['cw1', 'mid-term', 'cw2'])  # Tên các đặc trưng có thể thay đổi

# Vẽ biểu đồ so sánh kết quả dự đoán với kết quả thật
plot.plot_predictions(y_test, y_pred)