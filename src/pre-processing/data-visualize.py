import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ file CSV
data = pd.read_csv('Student-Final-Grade-ML/data/processed/grade-records_processed.csv')

# Vẽ histogram cho mỗi biến
data.hist(figsize=(10, 8), bins=30)
plt.tight_layout()
plt.show()

# Vẽ heatmap để xem mối tương quan
plt.figure(figsize=(10, 8))
sns.heatmap(data[['cw1', 'mid-term', 'cw2', 'final']].corr(), annot=True, cmap='coolwarm')
plt.title('Ma trận tương quan giữa các biến')
plt.show()

# Vẽ box plot cho các biến cụ thể
plt.figure(figsize=(12, 8))
sns.boxplot(data=data[['cw1', 'mid-term', 'cw2', 'final']])
plt.title('Box Plot cho các biến')
plt.show()