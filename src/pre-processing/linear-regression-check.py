import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
df = pd.read_csv('Student-Final-Grade-ML/data/processed/grade-records_processed.csv')

# Giả sử các cột 'cw1', 'mid-term', 'cw2' là đầu vào (X) và cột 'final' là đầu ra (y)
X = df[['cw1', 'mid-term', 'cw2']].values  # Lấy 3 cột làm input
y = df['final'].values.reshape(-1, 1)      # Lấy cột 'final' làm output

# Vẽ biểu đồ cho cw1 và final
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(X[:, 0], y, 'ro')
plt.xlabel('cw1 score')
plt.ylabel('Final score')

# Vẽ biểu đồ cho mid-term và final
plt.subplot(1, 3, 2)
plt.plot(X[:, 1], y, 'go')
plt.xlabel('Mid-term score')
plt.ylabel('Final score')

# Vẽ biểu đồ cho cw2 và final
plt.subplot(1, 3, 3)
plt.plot(X[:, 2], y, 'bo')
plt.xlabel('cw2 score')
plt.ylabel('Final score')

plt.tight_layout()
plt.show()

