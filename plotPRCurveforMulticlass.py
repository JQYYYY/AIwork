from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt
from numpy import interp
from sklearn.metrics import precision_recall_curve, average_precision_score


def plot_multiclass_PR(numClasses, yTest, yPredictProb, model_name):
    pre = dict()
    rec = dict()
    ap = dict()
    for i in range(numClasses):
        # pre[i], rec[i], _ = precision_recall_curve(yTest, yPredictProb[:, i], pos_label=i)
        pre[i], rec[i], _ = precision_recall_curve(yTest[:, i], yPredictProb[:, i])
    all_rec = np.unique(np.concatenate([rec[i] for i in range(numClasses)]))
    mean_pre = np.zeros_like(all_rec)
    for i in range(numClasses):
        mean_pre += interp(all_rec, pre[i], rec[i])
    mean_pre /= numClasses

    pre["macro"] = all_rec
    rec["macro"] = mean_pre
    ap["macro"] = average_precision_score(yTest, yPredictProb, average='macro')

    plt.plot(pre["macro"], rec["macro"], label=model_name + ' PR curve(MAP={})'.format(ap["macro"]), linewidth=1)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'deeppink', 'darkseagreen', 'gold', 'greenyellow',
                    'lightsalmon', 'maroon', 'mediumpurple'])

    # for i, color in zip(range(numClasses), colors):
    #       plt.plot(rec[i], pre[i], color=color, lw=1, label='PR curve of class {0}'.format(i))
    # if model_name == 'CNN':
    #     plt.xlabel('Recall')
    #     plt.ylabel('Precision')
    #     plt.title('Precision and Recall Curve')
    #     plt.legend(loc="lower right")
    #     plt.show()