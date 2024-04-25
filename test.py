import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# 假设我们有一个没有时间戳的观测值序列
values = [240, 250, 260, 270, 280, 290, 320, 310]

# 创建一个整数索引来代表时间
time_index = range(len(values))

# 将观测值转换为pandas Series，并设置索引
time_series_data = pd.Series(values, index=time_index)

# 初始化ARIMA模型
model = ARIMA(time_series_data, order=(1, 1, 1))

# 拟合模型
model_fit = model.fit()

# 进行预测
forecast = model_fit.predict(start=len(time_series_data), end=len(time_series_data) + 4)

# 打印预测结果
print(forecast.to_numpy())

# 可视化原始数据和预测结果
plt.figure(figsize=(10, 6))
plt.plot(time_series_data, label="Original Data")
plt.plot(forecast, label="Forecast", color="red")
plt.legend()
plt.show()
