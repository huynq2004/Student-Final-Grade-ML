import numpy as np
import cart_model as model
import cart_utils as utils
import cart_plots as plot

# 1. Thiết lập độ sâu cây cần kiểm tra
depth_values = range(1, 11)

# 2. Huấn luyện mô hình và tìm độ sâu tối ưu
errors = []
best_depth = None
lowest_error = float('inf')

for depth in depth_values:
    fold_errors = []
    for fold in range(5):
        X_train, y_train, X_val, y_val = utils.load_fold_data(fold)
        
        # Khởi tạo và huấn luyện Decision Tree
        tree = model.DecisionTree(max_depth=depth)
        tree.fit(X_train, y_train)
        
        # Dự đoán trên tập val
        y_pred = tree.predict(X_val)
        
        # Tính lỗi MSE
        mse = np.mean((y_val - y_pred) ** 2)
        fold_errors.append(mse)
    
    avg_error = np.mean(fold_errors)
    errors.append(avg_error)

    if avg_error < lowest_error:
        lowest_error = avg_error
        best_depth = depth

    print(f"Depth = {depth}, Average Fold Error = {avg_error}")

# 3. Lưu trữ mô hình với độ sâu tốt nhất
print(f'Độ sâu tối ưu (thủ công): {best_depth}')

# 4. Dự đoán trên tập test
X_test, y_test = utils.load_data_from('data/split/test_data.csv')
tree = model.DecisionTree(max_depth=best_depth)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

# 5. Tính lỗi và vẽ biểu đồ
mse_test = np.mean((y_test - y_pred) ** 2)
print(f'Lỗi trung bình trên tập test (thủ công): {mse_test}')

# Vẽ biểu đồ lỗi theo độ sâu
plot.plot_errors(depth_values, errors)
