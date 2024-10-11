import matplotlib.pyplot as plt

def plot_tree(tree, depth=0):
    # Implement tree visualization logic here if needed
    pass

def plot_errors(depths, errors):
    plt.plot(depths, errors)
    plt.xlabel('Tree Depth')
    plt.ylabel('Mean Squared Error')
    plt.title('Error by Tree Depth')
    plt.show()
