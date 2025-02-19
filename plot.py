import pandas as pd

import matplotlib.pyplot as plt

# CSVファイルを読み込む
data = pd.read_csv('log_accuracy.csv')

# noize_ratioとaccuracyのデータを取得
noize_ratio = data['noise_ratio']
accuracy = data['accuracy']

# プロットを作成
plt.figure()
plt.xlim(0, 0.25)
plt.ylim(0, 1)
# plt.plot(noize_ratio, accuracy, marker='o', linestyle='-', color='b')
plt.scatter(noize_ratio, accuracy, color='b')
plt.xlabel('Noize Ratio')
plt.ylabel('Accuracy')
plt.title('Noize Ratio vs Accuracy')
plt.grid(True)
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)

a = ax.plot([1, 2, 3])

print(a)

b = ax.plot()

print(b)