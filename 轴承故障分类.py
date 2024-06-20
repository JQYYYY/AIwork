import numpy as np
import pandas as pd
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, LSTM, BatchNormalization, Activation, Flatten
from keras.regularizers import l2
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

import plotAUCforMulticlass
import plotPRCurveforMulticlass

dataSet = pd.read_csv('Train.csv')
xData = dataSet.iloc[:, 1:-1]
yData = dataSet.iloc[:, -1]

scalar = preprocessing.StandardScaler()
xData = scalar.fit_transform(xData)
yData = np.array(yData).reshape([-1, 1])            # 转化为1列

numClasses = 10
batchSize = 128
epochs = 20         # 循环周期
batchSize1 = 16    # 一批次的大小
BatchNormal = True  # 是否批量归一化

# 模型
def CNN_LSTM():
    model = Sequential()

    model.add(Conv1D(filters=16, kernel_size=64, padding='same', input_shape=(xTrain.shape[1], 1)))
    model.add(Conv1D(filters=32, kernel_size=32, padding='same'))
    model.add(MaxPooling1D(60))

    model.add(Conv1D(filters=64, kernel_size=3, padding='same'))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(filters=256, kernel_size=3, padding='same'))
    model.add(MaxPooling1D(5))

    model.add(Conv1D(filters=512, kernel_size=3, padding='same'))
    model.add(MaxPooling1D(1))

    model.add(Dense(256))
    model.add(Dropout(0.3))

    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(16))

    model.add(Dense(numClasses, activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def wdcnn(filters, kernel_size, strides, conv_padding, pool_padding, pool_size, BatchNormal, model):
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=conv_padding,
                     kernel_regularizer=l2(1e-4)))
    if BatchNormal:
        model.add(BatchNormalization())
    model.add(Activation('relu'))       # relu层
    model.add(MaxPooling1D(pool_size=pool_size, padding=pool_padding))      # 池化层
    return model


def CNN():
    model = Sequential()  # 实例化序贯模型
    model.add(Conv1D(filters=16, kernel_size=64, strides=16, padding='same', kernel_regularizer=l2(1e-4),
                     input_shape=(xTrain.shape[1], 1)))  # 第一层卷积,需指定input_shape
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    model = wdcnn(filters=32, kernel_size=3, strides=1, conv_padding='same', pool_padding='valid', pool_size=2,
                  BatchNormal=BatchNormal, model=model)  # 第二层
    model = wdcnn(filters=64, kernel_size=3, strides=1, conv_padding='same', pool_padding='valid', pool_size=2,
                  BatchNormal=BatchNormal, model=model)
    model = wdcnn(filters=64, kernel_size=3, strides=1, conv_padding='same', pool_padding='valid', pool_size=2,
                  BatchNormal=BatchNormal, model=model)
    model = wdcnn(filters=64, kernel_size=3, strides=1, conv_padding='same', pool_padding='valid', pool_size=2,
                  BatchNormal=BatchNormal, model=model)

    model.add(Flatten())  # 从卷积到全连接展平
    model.add(Dense(units=100, activation='relu', kernel_regularizer=l2(1e-4)))  # 添加全连接层
    model.add(Dense(units=numClasses, activation='softmax', kernel_regularizer=l2(1e-4)))  # 添加输出层
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])  # 编译模型
    return model


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv = skf.get_n_splits(xData, yData)
for i, (train_index, test_index) in enumerate(skf.split(xData, yData)):
    print(f"第{i}折：\n")
    xTrain = xData[np.array(train_index).T]
    yTrain = yData[np.array(train_index).T]
    xTest = xData[np.array(test_index).T]
    yTest = yData[np.array(test_index).T]
    xTrain, xTest = xTrain[:, :, np.newaxis], xTest[:, :, np.newaxis]  # 输入卷积的时候还需要修改一下，增加通道数目
    Encoder = preprocessing.OneHotEncoder()
    yTest = Encoder.fit_transform(yTest).toarray()
    yTest = np.asarray(yTest, dtype=np.int32)
    yTrain = Encoder.fit_transform(yTrain).toarray()
    yTrain = np.asarray(yTrain, dtype=np.int32)

    tb_cb = TensorBoard(log_dir='logs')  # 查看训练情况
    # CNN_LSTM
    model = CNN_LSTM()
    model1 = CNN()
    H = model.fit(x=xTrain, y=yTrain, batch_size=batchSize, epochs=epochs, verbose=1, shuffle=True,
                  callbacks=[tb_cb])  # 开始模型训练
    H1 = model1.fit(x=xTrain, y=yTrain, batch_size=batchSize1, epochs=epochs, verbose=1, shuffle=True,
                    callbacks=[tb_cb])
    score = model.evaluate(x=xTest, y=yTest, verbose=0)  # 训练结果
    score1 = model1.evaluate(x=xTest, y=yTest, verbose=0)
    print("CNN-LSTM测试集上的损失：", score[0])
    print("CNN-LSTM测试集上的准确度：", score[1])
    print("CNN测试集上的损失：", score1[0])
    print("CNN测试集上的准确率：", score1[1])

    yPredict = model.predict(xTest)
    yPredict1 = model1.predict(xTest)

    font = {'family': 'Times New Roman'}
    plt.rc('font', **font)

    plt.figure()
    plotAUCforMulticlass.plot_multiclass_AUC(numClasses, yTest, yPredict, 'CNN-LSTM')
    plotAUCforMulticlass.plot_multiclass_AUC(numClasses=numClasses, yTest=yTest, yPredict=yPredict1, model_name='CNN')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    plt.figure()
    plotPRCurveforMulticlass.plot_multiclass_PR(numClasses, yTest, yPredict, 'CNN-LSTM')
    plotPRCurveforMulticlass.plot_multiclass_PR(numClasses, yTest, yPredict1, model_name='CNN')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision and Recall Curve')
    plt.legend(loc="lower right")
    plt.show()