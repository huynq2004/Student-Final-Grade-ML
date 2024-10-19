from graphviz import Digraph
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree

def plot_decision_tree(model, feature_names=None):
    """
    Vẽ cây quyết định sau khi huấn luyện.
    
    Parameters:
    - model: Mô hình DecisionTreeRegressor đã huấn luyện.
    - feature_names: Danh sách tên các đặc trưng (nếu có).
    """
    plt.figure(figsize=(20, 10))  # Thiết lập kích thước đồ thị
    tree.plot_tree(model, feature_names=feature_names, filled=True, rounded=True, fontsize=10)
    plt.show()

def plot_tree(tree, feature_names=None):
    """
    Hàm vẽ cây quyết định dựa trên cấu trúc cây được huấn luyện.
    
    Args:
        tree: Cây quyết định dưới dạng dict, với mỗi node là một dict chứa
              feature, threshold, left, right. Lá là một số float (giá trị dự đoán).
        feature_names: Danh sách tên các thuộc tính (feature) để hiển thị.
    """
    def recurse(node, parent_name='', graph=None, node_id=0):
        # Tạo đồ thị nếu chưa có
        if graph is None:
            graph = Digraph()
            graph.node(f'{node_id}', 'Root')
        
        # Tăng node_id để đảm bảo mỗi node có tên duy nhất
        current_node_id = node_id

        if isinstance(node, dict):  # Nếu node là một dict (không phải lá)
            feature = feature_names[node['feature']] if feature_names else f'X[{node["feature"]}]'
            threshold = node['threshold']
            node_label = f'{feature} <= {threshold:.2f}'
            
            left_child_id = current_node_id + 1
            right_child_id = left_child_id + 1

            # Thêm node hiện tại vào cây
            graph.node(f'{left_child_id}', label=f'{feature} <= {threshold:.2f}')
            graph.node(f'{right_child_id}', label=f'{feature} > {threshold:.2f}')
            
            # Kết nối node hiện tại với các nhánh
            graph.edge(f'{current_node_id}', f'{left_child_id}')
            graph.edge(f'{current_node_id}', f'{right_child_id}')
            
            # Đệ quy vẽ các nhánh trái và phải
            graph = recurse(node['left'], parent_name=f'left_{parent_name}', graph=graph, node_id=left_child_id)
            graph = recurse(node['right'], parent_name=f'right_{parent_name}', graph=graph, node_id=right_child_id)
        
        else:  # Nếu là node lá (là một số dự đoán)
            leaf_node_id = current_node_id + 1
            graph.node(f'{leaf_node_id}', label=f'Predict: {float(node):.2f}', shape='box')
            graph.edge(f'{current_node_id}', f'{leaf_node_id}')
        
        return graph

    # Bắt đầu vẽ từ gốc
    graph = recurse(tree, parent_name='0')
    graph.render("decision_tree", "results/plots/decision-tree-CART" ,format="png", cleanup=True)  # Lưu ảnh cây dưới dạng PNG
    print("Cây quyết định đã được lưu thành file decision_tree.png")

def plot_errors(depths, errors):
    plt.plot(depths, errors)
    plt.xlabel('Giá trị độ sâu')
    plt.ylabel('Lỗi trung bình ')
    plt.title('Lỗi theo các giá trị độ sâu')
    plt.show()


