import pickle

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

numClass = 3
classLabel = ['high', 'medium', 'low']
sheet0 = pd.read_excel(io='C:\\Users\\86133\\PycharmProjects\\artifiacial intelligence\\声诊练习.xlsx', sheet_name=0, usecols="C:L")       # sheet_name=None读取全部sheet
sheet0.loc[:, 'label'] = 0          # 列增加label
sheet0.iloc[:, 0:-1] = np.sqrt(sheet0.iloc[:, 0:-1])
sheet1 = pd.read_excel(io='C:\\Users\\86133\\PycharmProjects\\artifiacial intelligence\\声诊练习.xlsx', sheet_name=1, usecols="C:L")
sheet1.loc[:, 'label'] = 1
sheet1.iloc[:, 0:-1] = np.sqrt(sheet1.iloc[:, 0:-1])
sheet2 = pd.read_excel(io='C:\\Users\\86133\\PycharmProjects\\artifiacial intelligence\\声诊练习.xlsx', sheet_name=2, usecols="C:L")
sheet2.loc[:, 'label'] = 2
sheet2.iloc[:, 0:-1] = np.sqrt(sheet2.iloc[:, 0:-1])
sheets = pd.concat([sheet0, sheet1, sheet2])      # 按行合并三张表

x = sheets.iloc[:, 0:-1]
y = sheets.iloc[:, -1]             # 获得标签
y = np.array(y)
transfer = StandardScaler()     # 标准化
x = transfer.fit_transform(x)   # 划分折数

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=23)
estimator = XGBClassifier(gamma=1, objective='multi:softmax', num_class=3,
                          learning_rate=1, n_estimators=200, reg_alpha=1, reg_lambda=1)
# estimator = RandomForestClassifier(criterion='log_loss', max_features='sqrt', n_estimators=100, random_state=10)
estimator.fit(x_train, y_train)
score = estimator.score(x_test, y_test)
print(f"score:{score}")

font = {'family': 'Times New Roman'}
plt.rc('font', **font)

# 设置特征名称列表
feature_names = ['F0', 'I', 'F1', 'F2', 'F3', 'F4', 'B1', 'B2', 'B3', 'B4']

# 特征贡献图
# explainer = shap.KernelExplainer(estimator.predict, x_train)
# shap_values = explainer.shap_values(x_train)
# 摘要图
# shap.summary_plot(shap_values, x_train, feature_names=feature_names)
# plt.show()
# 条形图
# shap.summary_plot(shap_values, x_train, plot_type="bar", feature_names=feature_names)
# plt.show()



# 混淆矩阵
# yPredict = estimator.predict(x_test)
# cm = confusion_matrix(y_test, yPredict)
# cm_display = ConfusionMatrixDisplay(cm).plot()
# plt.show()