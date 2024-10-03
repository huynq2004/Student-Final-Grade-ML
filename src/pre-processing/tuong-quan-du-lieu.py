import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# Đọc dữ liệu từ file CSV
data = pd.read_csv('Student-Final-Grade-ML/data/processed/grade-records_processed.csv')

# Vẽ heatmap để xem mối tương quan
plt.figure(figsize=(10, 8))
sns.heatmap(data[['cw1', 'mid-term', 'cw2', 'final']].corr(), annot=True, cmap='coolwarm')
plt.title('Ma trận tương quan giữa các biến (sau khi chuyển đổi)')
plt.show()