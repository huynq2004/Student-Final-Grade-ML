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

#   Biểu đồ phân tán so sánh final real vs final pridict trên test  #
def plot_final_scores_comparison(X_test, y_test, weights):
    y_pred = model.predict(X_test, weights)    # Dự đoán điểm final
    # Vẽ biểu đồ
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, y_pred, color='blue', label='Dự đoán', alpha=0.7)
    # Vẽ đường thẳng y = x để so sánh
    x = np.linspace(0, 10, 100)
    plt.plot(x, x, color='red', linestyle='--', label='Giá trị thực')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel('Giá trị thực của final')
    plt.ylabel('Giá trị dự đoán của final')
    plt.title('So sánh điểm final dự đoán và thực tế trên tập test')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid()
    plt.legend()
    plt.show()