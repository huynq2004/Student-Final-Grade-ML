import matplotlib.pyplot as plt

def plot_tree(tree, depth=0):
    # Implement tree visualization logic here if needed
    pass

def plot_errors(depths, errors):
    plt.plot(depths, errors)
    plt.xlabel('Giá trị độ sâu')
    plt.ylabel('Lỗi trung bình ')
    plt.title('Lỗi theo các giá trị độ sâu')
    plt.show()
    
