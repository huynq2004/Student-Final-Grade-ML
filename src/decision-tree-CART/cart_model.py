import numpy as np

# Hàm xây dựng cây quyết định (CART)
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    # Hàm tính độ lệch Gini hoặc MSE cho bài toán regression
    def _calculate_split_error(self, y):
        # Với bài toán regression, chúng ta có thể dùng Mean Squared Error (MSE)
        return np.mean((y - np.mean(y)) ** 2)

    # Hàm tìm split tốt nhất dựa trên các thuộc tính và nhãn
    def _best_split(self, X, y):
        best_split = {}
        best_error = float('inf')
        
        # Thử từng thuộc tính để tìm split tốt nhất
        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold
                
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue
                
                # Tính lỗi cho hai nhánh (left, right)
                left_error = self._calculate_split_error(y[left_indices])
                right_error = self._calculate_split_error(y[right_indices])
                split_error = (np.sum(left_indices) * left_error + np.sum(right_indices) * right_error) / len(y)

                # Lưu lại split có lỗi nhỏ nhất
                if split_error < best_error:
                    best_error = split_error
                    best_split = {
                        'feature': feature,
                        'threshold': threshold,
                        'left_indices': left_indices,
                        'right_indices': right_indices
                    }
        
        return best_split

    # Hàm xây dựng cây
    def _build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or depth == self.max_depth:
            return np.mean(y)  # Giá trị dự đoán tại lá

        split = self._best_split(X, y)
        if split == {}:
            return np.mean(y)

        left_tree = self._build_tree(X[split['left_indices']], y[split['left_indices']], depth + 1)
        right_tree = self._build_tree(X[split['right_indices']], y[split['right_indices']], depth + 1)

        return {'feature': split['feature'], 'threshold': split['threshold'], 'left': left_tree, 'right': right_tree}

    # Hàm huấn luyện mô hình
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    # Hàm dự đoán với cây đã huấn luyện
    def predict_sample(self, x, tree):
        if isinstance(tree, dict):
            feature = tree['feature']
            threshold = tree['threshold']
            if x[feature] <= threshold:
                return self.predict_sample(x, tree['left'])
            else:
                return self.predict_sample(x, tree['right'])
        else:
            return tree

    def predict(self, X):
        return np.array([self.predict_sample(x, self.tree) for x in X])

