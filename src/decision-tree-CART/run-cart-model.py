import numpy as np
import cart_model as model
import cart_utils as utils
import cart_plots as plot
import pickle
# 1. Thiết lập độ sâu cây cần kiểm tra

depth_values = range(1, 11)
threshold = 0.01  


# 2. Huấn luyện mô hình và tìm độ sâu tối ưu
errors = []
best_depth = None
lowest_error = float('inf')
previous_error = float('inf')  # Khởi tạo lỗi ban đầu

for depth in depth_values:
    fold_errors = []
    for fold in range(5):
        X_train, y_train, X_val, y_val = utils.load_fold_data(fold)
        tree = model.DecisionTree(max_depth=depth)
        tree.fit(X_train, y_train)

        y_pred = tree.predict(X_val)
        # Tính lỗi MSE
        mse = np.mean((y_val - y_pred) ** 2)
        fold_errors.append(mse)
    
    avg_error = np.mean(fold_errors)
    errors.append(avg_error)
    # So sánh lỗi trung bình với ngưỡng dừng sớm
    if abs(previous_error - avg_error) < threshold:
        print(f"Sự khác biệt lỗi giữa độ sâu {depth-1} và {depth} nhỏ hơn ngưỡng {threshold}. Dừng sớm.")
        break
    
    # Cập nhật giá trị lỗi và độ sâu tốt nhất
    if avg_error < lowest_error:
        lowest_error = avg_error
        best_depth = depth
        
    previous_error = avg_error  # Cập nhật lỗi của độ sâu trước đó
    print(f"Depth = {depth}, Average Fold Error = {avg_error}")

# 3. Lưu trữ mô hình với độ sâu tốt nhất
print(f'Độ sâu tối ưu (thủ công): {best_depth}')
best_tree = model.DecisionTree(max_depth=best_depth)
X_train, y_train = utils.load_data_from('data/split/train_data.csv')
best_tree.fit(X_train, y_train)

with open('models/cart_model_best_depth.pkl', 'wb') as file:
    pickle.dump(best_tree, file)


with open('models/cart_model_best_depth.pkl', 'rb') as file:
    loaded_tree = pickle.load(file)

# 4. Dự đoán trên tập test
X_test, y_test = utils.load_data_from('data/split/test_data.csv')
y_pred = loaded_tree.predict(X_test)

# 5. Tính lỗi và vẽ biểu đồ
mse_test = np.mean((y_test - y_pred) ** 2)
print(f'Lỗi trung bình trên tập test (thủ công): {mse_test}')

# Đặt ngưỡng sai số cho accuracy
error_threshold = 0.5  

# Tính accuracy
correct_predictions = np.sum(np.abs(y_test - y_pred) < error_threshold)
accuracy = (correct_predictions / len(y_test) ) 

print(f'Accuracy (ngưỡng {error_threshold}): {accuracy:.2f}%')


# Vẽ biểu đồ lỗi theo độ sâu
plot.plot_errors(depth_values[:len(errors)], errors)

# Vẽ cây kết quả
plot.plot_tree(tree.tree, feature_names=['cw1', 'mid-term', 'cw2'])

# Vẽ biểu đồ so sánh kết quả dự đoán với kết quả thật
plot.plot_predictions(y_test, y_pred)