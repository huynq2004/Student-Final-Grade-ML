import matplotlib.pyplot as plt
import numpy as np
import ridge_utils as utils
import ridge_model as model

def plot_errors(lamda_values, errors):
    plt.plot(lamda_values, errors)
    plt.xlabel('Giá trị λ')
    plt.ylabel('Lỗi trung bình')
    plt.title('Lỗi theo các giá trị λ')
    plt.show()

def plot_final_scores_comparison(X_test, y_test, y_pred):
    # Vẽ biểu đồ phân tán
    plt.figure(figsize=(6, 6))  # Điều chỉnh kích thước biểu đồ (bé hơn)

    plt.scatter(y_test, y_pred, color='blue', label='Dự đoán', alpha=0.7)
    
    # Vẽ đường thẳng y = x để so sánh
    x = np.linspace(0, 10, 100)
    plt.plot(x, x, color='red', linestyle='--', label='Regression Line')

    # Giới hạn trục từ 0 đến 10
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    # Thêm các thông số của biểu đồ
    plt.xlabel('Giá trị nhãn thực tế')
    plt.ylabel('Giá trị nhãn dự đoán ')
    plt.title('So sánh nhãn thực tế và nhãn lý thuyết')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid(True)
    plt.legend()
    
    # Hiển thị biểu đồ
    plt.show()