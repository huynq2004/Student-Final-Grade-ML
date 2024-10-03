import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


# Đọc dữ liệu từ file CSV
data = pd.read_csv('Student-Final-Grade-ML/data/processed/grade-records_processed.csv')

# Vẽ histogram cho mỗi biến
data.hist(figsize=(10, 8), bins=30)
plt.tight_layout()
plt.show()