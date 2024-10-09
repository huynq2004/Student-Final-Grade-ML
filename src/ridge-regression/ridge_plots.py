import matplotlib.pyplot as plt

def plot_errors(lamda_values, errors):
    plt.plot(lamda_values, errors)
    plt.xlabel('Giá trị λ')
    plt.ylabel('Lỗi trung bình')
    plt.title('Lỗi theo các giá trị λ')
    plt.show()
