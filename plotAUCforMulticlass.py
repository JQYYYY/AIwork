from itertools import cycle
import numpy as np
from matplotlib import pyplot as plt
from numpy import interp
from sklearn.metrics import roc_curve, auc

def plot_multiclass_AUC(numClasses, yTest, yPredict, model_name):
    """画出多分类结果的AUC，右下角为不同种类为正例时的AUC
    :param numClasses: 多分类的种数
    :param yTest: 测试样本的真实结果，无需是二值结果，但需要是数值型
    :param yPredict: 测试样本的预测结果
    :return: 无
    """

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(numClasses):          # 遍历类别
        fpr[i], tpr[i], _ = roc_curve(yTest[:, i], yPredict[:, i])          # 分别计算每个类别为正例时的fpr、tpr
        roc_auc[i] = auc(fpr[i], tpr[i])                                    # 分别计算每个类别为正例时的AUC（面积）
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(numClasses)]))    # 去除重复的fpr，以便插值
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(numClasses):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])                         # 线性插值，求得平均的tpr
    mean_tpr /= numClasses

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.plot(fpr["macro"], tpr["macro"],
             label=model_name + ' ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]), linewidth=1)              # 平均曲线

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'deeppink', 'darkseagreen', 'gold',  'greenyellow',
                    'lightsalmon', 'maroon', 'mediumpurple'])
    # for i, color in zip(range(numClasses), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #                    ''.format(i, roc_auc[i]))
    # if model_name == 'CNN':
    #     plt.plot([0, 1], [0, 1], 'k--', lw=1)
    #     plt.xlim([0.0, 1.05])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Some extension of Receiver operating characteristic to multi-class')
    #     plt.legend(loc="lower right")
        # plt.show()