import pickle
from itertools import cycle

import numpy as np
import pandas as pd
import seaborn
import shap
from matplotlib import pyplot as plt
from numpy import interp
from pandas.plotting import parallel_coordinates
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score, classification_report, \
    precision_score, recall_score, log_loss, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.svm import SVC
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

# 画出数据属性的平行坐标图
# parallel_coordinates(sheets, 'label', color=('deeppink', 'cornflowerblue', '#C7F464'))
# plt.show()

x = sheets.iloc[:, 0:-1]

# 主成分提取
# model = decomposition.PCA(n_components=7)
# model.fit(x)
# x = model.transform(x)
# print(f"保留主成分的贡献率:{model.explained_variance_ratio_}")
# print(f"具有最大方差的成分:{model.components_}

# 相关系数矩阵
# corr_matrix = x.corr()

# 热力图
# seaborn.heatmap(corr_matrix, annot=True)
# plt.show()

y = sheets.iloc[:, -1]             # 获得标签
y = np.array(y)
transfer = StandardScaler()     # 标准化
x = transfer.fit_transform(x)   # 划分折数

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv = skf.get_n_splits(x, y)


fprAverage = dict()
tprAverage = dict()
aucAverage = dict()
preAverage = dict()
recAverage = dict()
apAverage = dict()

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'deeppink', 'darkseagreen', 'gold', 'greenyellow',
                'lightsalmon', 'maroon', 'mediumpurple'])

allScore = []
auc_list = []
pre_list = []
rec_list = []
loss_list = []
ap_list = []

for i, (train_index, test_index) in enumerate(skf.split(x, y)):
    # print(f"第{i}折：\n")
    xTrain = x[np.array(train_index).T]
    yTrain = y[np.array(train_index).T]
    xTest = x[np.array(test_index).T]
    yTest = y[np.array(test_index).T]

    # 转化为onehot编码
    label_binarizer = LabelBinarizer().fit(yTrain)
    y_onehot_test = label_binarizer.transform(yTest)

    # estimator = XGBClassifier(gamma=1, objective='multi:softmax', num_class=3,
    #                           learning_rate=1, n_estimators=200, reg_alpha=1, reg_lambda=1)
    # estimator = SVC(C=1, kernel='rbf', gamma=0.1, probability=True)
    estimator = RandomForestClassifier(criterion='log_loss', max_features='sqrt', n_estimators=100, random_state=10)
    estimator.fit(xTrain, yTrain)

    score = estimator.score(xTest, yTest)
    allScore.append(score)

    yPredictProb = estimator.predict_proba(xTest)
    yPredict = estimator.predict(xTest)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    pre = dict()
    rec = dict()
    for j in range(numClass):  # 遍历类别
        fpr[j], tpr[j], _ = roc_curve(y_onehot_test[:, j], yPredictProb[:, j])  # 分别计算每个类别为正例时的fpr、tpr

    # 计算每折macro-average
    pre_list.append(precision_score(yTest, yPredict, average='macro'))
    rec_list.append(recall_score(yTest, yPredict, average='macro'))

    # 计算每折macro-average
    for j in range(numClass):
        fpr[j], tpr[j], _ = roc_curve(y_onehot_test[:, j], yPredictProb[:, j])
        roc_auc[j] = auc(fpr[j], tpr[j])
        pre[j], rec[j], _ = precision_recall_curve(y_onehot_test[:, j], yPredictProb[:, j])
    fprAverage[i] = np.unique(np.concatenate([fpr[i] for i in range(numClass)]))
    recAverage[i] = np.unique(np.concatenate([rec[i] for i in range(numClass)]))
    tprAverage[i] = np.zeros_like(fprAverage[i])
    preAverage[i] = np.zeros_like(recAverage[i])
    for j in range(numClass):
        tprAverage[i] += interp(fprAverage[i], fpr[j], tpr[j])  # 线性插值，求得平均的tpr
        preAverage[i] += interp(recAverage[i], np.flipud(rec[j]), np.flipud(pre[j]))
    tprAverage[i] /= numClass
    preAverage[i] /= numClass
    aucAverage[i] = auc(fprAverage[i], tprAverage[i])
    apAverage[i] = average_precision_score(y_onehot_test, yPredictProb, average="macro")

    auc_list.append(aucAverage[i])  # 储存每一折的auc
    loss_list.append(log_loss(yTest, yPredictProb))
    ap_list.append(apAverage[i])

    # 查看当前折Precision、recall
    report = classification_report(yTest, yPredict, labels=[0, 1, 2], target_names=['high', 'medium', 'low'])
    print(report)

    # summary_plot
    # explainer = shap.KernelExplainer(estimator.predict, xTest)
    # shap_values = explainer.shap_values(xTest)
    # shap.summary_plot(shap_values, xTest)

    # distances, indices = estimator.kneighbors(xTest)
    # neighbor_indices = indices[0]
    # for i, index in enumerate(neighbor_indices):
    #     neighbor_sample = xTrain[index]
    #     neighbor_label = yTrain[index]
    #     print(f"Neighbor {i + 1}:{neighbor_sample}(Label:{neighbor_label})")

    # 混淆矩阵
    # plt.figure(i)
    # yPredict = estimator.predict(xTest)
    # cm = confusion_matrix(yTest, yPredict)
    # cm_display = ConfusionMatrixDisplay(cm).plot()

# 所有折的平均fpr和tpr----------------------------------------------------------
all_mean_fpr = np.unique(np.concatenate([fprAverage[i] for i in range(cv)]))
all_mean_tpr = np.zeros_like(all_mean_fpr)
all_mean_rec = np.unique(np.concatenate([recAverage[i] for i in range(cv)]))
all_mean_pre = np.zeros_like(all_mean_rec)

for k in range(cv):
    all_mean_tpr += interp(all_mean_fpr, fprAverage[k], tprAverage[k])
    all_mean_pre += interp(all_mean_rec, recAverage[k], preAverage[k])
all_mean_tpr /= cv
all_mean_pre /= cv
fprAverage['average'] = all_mean_fpr
tprAverage['average'] = all_mean_tpr
aucAverage['average'] = auc(all_mean_fpr, all_mean_tpr)
recAverage['average'] = all_mean_rec
preAverage['average'] = all_mean_pre
apAverage['average'] = sum(apAverage[i] for i in range(cv)) / cv

print(f"每折得分：{allScore}\n 平均得分：{sum(allScore)/cv}")
print(f"每折AUC：{auc_list}\n平均AUC：{sum(auc_list) / cv}")
print(f"平均Precision：{sum(pre_list) / cv}")
print(f"平均Recall:{sum(rec_list) / cv}")
print(f"平均AP：{sum(ap_list) / cv}")
print(f"损失:{sum(loss_list) / cv}\n")

# 保存模型
# s = pickle.dumps(estimator)
# f = open('xgboost.model', 'wb+')
# f.write(s)
# f.close()
# print("Done")

# font = {'family': 'Times New Roman'}
# plt.rc('font', **font)
#
# plt.figure(6)
# plt.plot(fprAverage['average'], tprAverage['average'],
#          label='average ROC curve of all folds (area = {0:0.2f})'
#          .format(aucAverage['average']),
#          color='navy', linestyle=':', linewidth=2)
#
# for i, color in zip(range(cv), colors):
#     plt.plot(fprAverage[i], tprAverage[i], color=color, lw=0.5,
#              label='micro-average ROC curve of fold {0} (area = {1:0.2f})'
#                    ''.format(i, aucAverage[i]))
#
# plt.plot([0, 1], [0, 1], 'k--', lw=1)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC curve')
# legend = plt.legend(loc="lower right")
# legend.get_frame().set_alpha(0.3)
# plt.show()
#
# plt.figure(7)
# plt.plot(recAverage['average'], preAverage['average'],
#          label='average ROC curve of all folds (mAP={0:0.2f})'
#          .format(apAverage['average']),
#          color='navy', linestyle=':', linewidth=2)
#
# for i, color in zip(range(cv), colors):
#     plt.plot(recAverage[i], preAverage[i], color=color, lw=0.5,
#              label='micro-average PR curve of fold {0}(AP={1:0.2f})'
#                    ''.format(i, apAverage[i]))
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('PR curve')
# legend = plt.legend(loc="best")
# legend.get_frame().set_alpha(0.3)
# plt.show()