import warnings
from itertools import cycle

import pandas as pd
import numpy as np
import shap.explainers
from numpy import interp
from pandas.plotting import parallel_coordinates
from sklearn import preprocessing, decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import StratifiedKFold, ParameterGrid, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, log_loss, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, \
    average_precision_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, PrecisionRecallDisplay
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from xgboost import XGBClassifier

import getBestParam


warnings.filterwarnings("ignore")
numClass = 3
classLabel = ['high', 'medium', 'low']
sheet0 = pd.read_excel(io='C:\\Users\\86133\\PycharmProjects\\artifiacial intelligence\\声诊练习.xlsx', sheet_name=0, usecols="C:L")       # sheet_name=None读取全部sheet
sheet0.loc[:, 'label'] = 0                                             # 列增加label
sheet1 = pd.read_excel(io='C:\\Users\\86133\\PycharmProjects\\artifiacial intelligence\\声诊练习.xlsx', sheet_name=1, usecols="C:L")
sheet1.loc[:, 'label'] = 1
sheet2 = pd.read_excel(io='C:\\Users\\86133\\PycharmProjects\\artifiacial intelligence\\声诊练习.xlsx', sheet_name=2, usecols="C:L")
sheet2.loc[:, 'label'] = 2
sheets = pd.concat([sheet0, sheet1, sheet2])      # 按行合并三张表

x = sheets.iloc[:, 0:-1]
y = sheets.iloc[:, -1]             # 获得标签
y = np.array(y)

# 主成分提取
# model = decomposition.PCA(n_components=6)
# model.fit(x)
# x = model.transform(x)
# print(f"保留主成分的贡献率:{model.explained_variance_ratio_}")
# print(f"具有最大方差的成分:{model.components_}")

transfer = StandardScaler()     # 标准化
x = transfer.fit_transform(x)   # 划分折数

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv = skf.get_n_splits(x, y)

# KNN, LR, RF, SVM, bayes, xgboost
meansDict = {0: 'KNN', 1: 'LR', 2: 'RF', 3: 'SVM', 4: 'xgboost', 5: 'bayes'}

estimatorList = [KNeighborsClassifier(n_neighbors=6, weights='distance', algorithm='auto', p=1, metric='manhattan'),
                 LogisticRegression(C=0.5, max_iter=500, solver='lbfgs', multi_class='ovr'),
                 RandomForestClassifier(criterion='log_loss', max_features='sqrt', n_estimators=100, random_state=10),
                 SVC(C=1, kernel='rbf', gamma=0.1, probability=True),
                 XGBClassifier(gamma=1, objective='multi:softmax', num_class=3,
                               learning_rate=1, n_estimators=200, reg_alpha=1, reg_lambda=1), GaussianNB()]

best_param_list = []      # 存储每种算法的最佳参数

fprAverage = dict()
tprAverage = dict()
aucAverage = dict()
preAverage = dict()
recAverage = dict()
apAverage = dict()
std_pre = []
std_rec = []
std_acc = []

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'deeppink', 'darkseagreen', 'gold', 'greenyellow',
                'lightsalmon', 'maroon', 'mediumpurple'])

font = {'family': 'Times New Roman'}
plt.rc('font', **font)


# 每一折
for i, (train_index, test_index) in enumerate(skf.split(x, y)):
    print(f"第{i}折：\n")
    xTrain = x[np.array(train_index).T]
    yTrain = y[np.array(train_index).T]
    xTest = x[np.array(test_index).T]
    yTest = y[np.array(test_index).T]

    # 每一种算法
    cnt = -1
    for estimator in estimatorList:
        cnt += 1    # 对应列表中算法所在下标
        print(f"{meansDict[cnt]}:\n")

        # 转化为onehot编码
        label_binarizer = LabelBinarizer().fit(yTrain)
        y_onehot_test = label_binarizer.transform(yTest)

        estimator.fit(xTrain, yTrain)
        # 准确率
        score = estimator.score(xTest, yTest)

        yPredictProb = estimator.predict_proba(xTest)
        yPredict = estimator.predict(xTest)

        # 每折的平均fpr和tpr
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        pre = dict()
        rec = dict()

        # 计算每折micro-average
        # fprAverage[cnt], tprAverage[cnt], _ = roc_curve(y_onehot_test.ravel(), yPredictProb.ravel())
        # preAverage[cnt], recAverage[cnt], _ = precision_recall_curve(y_onehot_test.ravel(), yPredictProb.ravel())
        #
        # pre = precision_score(yTest, yPredict, average='micro')
        # rec = recall_score(yTest, yPredict, average='micro')
        # aucAverage[cnt] = auc(fprAverage[cnt], tprAverage[cnt])
        # apAverage[cnt] = average_precision_score(y_onehot_test, yPredictProb, average='micro')

        # 计算当前折macro-average
        for j in range(numClass):
            fpr[j], tpr[j], _ = roc_curve(y_onehot_test[:, j], yPredictProb[:, j])
            pre[j], rec[j], _ = precision_recall_curve(y_onehot_test[:, j], yPredictProb[:, j])
        fprAverage[cnt] = np.unique(np.concatenate([fpr[i] for i in range(numClass)]))
        recAverage[cnt] = np.unique(np.concatenate([rec[i] for i in range(numClass)]))
        tprAverage[cnt] = np.zeros_like(fprAverage[cnt])
        preAverage[cnt] = np.zeros_like(recAverage[cnt])
        for j in range(numClass):
            tprAverage[cnt] += interp(fprAverage[cnt], fpr[j], tpr[j])  # 线性插值，求得平均的tpr
            preAverage[cnt] += interp(recAverage[cnt], np.flipud(rec[j]), np.flipud(pre[j]))
        tprAverage[cnt] /= numClass
        preAverage[cnt] /= numClass
        aucAverage[cnt] = roc_auc_score(y_onehot_test, yPredictProb, multi_class='ovr', average="macro")
        apAverage[cnt] = average_precision_score(y_onehot_test, yPredictProb, average="macro")

        rec1 = recall_score(yTest, yPredict, average='macro')
        pre1 = precision_score(yTest, yPredict, average='macro')
        loss = log_loss(yTest, yPredictProb)

        # 查看当前折Precision、recall
        report = classification_report(yTest, yPredict, labels=[0, 1, 2], target_names=['high', 'medium', 'low'])
        print(report)

        std_pre.append(pre1)
        std_rec.append(rec1)
        std_acc.append(score)
        print(f"score：{score}")
        print(f"AUC：{aucAverage[cnt]}")
        print(f"Precision：{pre1}")
        print(f"Recall:{rec1}")
        print(f"AP：{apAverage[cnt]}")
        print(f"loss:{loss}\n")

# 计算标准差
# std_pre = np.array(std_pre).reshape(5, 6)
# std_rec = np.array(std_rec).reshape(5, 6)
# std_acc = np.array(std_acc).reshape(5, 6)
# std_deviation = np.std(std_acc, axis=0, ddof=0)
# print(np.mean(std_acc, axis=0))
# print(std_deviation)

    # 绘制ROC曲线
    # plt.figure()
    # for k, color in zip(range(len(meansDict)), colors):      # len(meansdict)
    #     # coefficients = np.polyfit(fprAverage[k], tprAverage[k], 2)
    #     # poly = np.poly1d(coefficients)
    #     # smooth_fpr = np.linspace(0, 1, 100)
    #     # smooth_tpr = poly(smooth_fpr)
    #     plt.plot(fprAverage[k], tprAverage[k], color=color, lw=0.5,
    #              label='ROC curve of {0} (area = {1:0.2f})'
    #                    ''.format(meansDict[k], aucAverage[k]))
    # plt.plot([0, 1], [0, 1], 'k--', lw=1)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC curve')
    # legend = plt.legend(loc="best")
    # legend.get_frame().set_alpha(0.3)
    # plt.savefig('ROC of fold{0}.svg'.format(i))
    #
    # # 绘制PR曲线
    # plt.figure()
    # for k, color in zip(range(len(meansDict)), colors):
    #     # coefficients = np.polyfit(recAverage[k], preAverage[k], 2)
    #     # poly = np.poly1d(coefficients)
    #     # smooth_rec = np.linspace(0, 1, 100)
    #     # smooth_pre = poly(smooth_rec)
    #     plt.plot(recAverage[k], preAverage[k], color=color, lw=0.5,
    #              label='PR curve of {0}(AP={1:0.2f})'
    #                    ''.format(meansDict[k], apAverage[k]))
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('PR curve', family='Times New Roman')
    # legend = plt.legend(loc="best", prop=font)
    # legend.get_frame().set_alpha(0.3)
    # plt.savefig('PR of fold{0}.svg'.format(i))