import numpy as np
import pandas as pd
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.engine.saving import model_from_json
from keras.layers import Dense, LSTM, Dropout, Flatten, Conv1D, BatchNormalization, Activation, MaxPooling1D
from keras.regularizers import l2
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold
import plotAUCforMulticlass
import plotPRCurveforMulticlass

dataSet = pd.read_csv('Train.csv')
xData = dataSet.iloc[:, 1:-1]
yData = dataSet.iloc[:, -1]
testSet = pd.read_csv('Test_data.csv')              # 获得测试集
testData = testSet.iloc[:, 1:]

scalar = preprocessing.StandardScaler()
xData = scalar.fit_transform(xData)
testData = scalar.fit_transform(testData)

yData = np.array(yData).reshape([-1, 1])            # 转化为1列
# Encoder = preprocessing.OneHotEncoder()             # 独热编码
# Encoder.fit(yData)
# yData = Encoder.transform(yData).toarray()
# yData = np.asarray(yData, dtype=np.int32)

# testSet = pd.read_csv('Test_data.csv')
# test = testSet.iloc[:, 0:-1]   # 需要分类的数据集
# test = np.array(test)
# test = test[:, :, np.newaxis]

xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.2, random_state=42)
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv = skf.get_n_splits(xData, yData)

# for i, (train_index, test_index) in enumerate(skf.split(xData, yData)):
#     print(f"第{i}折：\n")
#     xTrain = xData[np.array(train_index).T]
#     yTrain = yData[np.array(train_index).T]
#     xTest = xData[np.array(test_index).T]
#     yTest = yData[np.array(test_index).T]
xTrain, xTest = xTrain[:, :, np.newaxis], xTest[:, :, np.newaxis]    # 输入卷积的时候还需要修改一下，增加通道数目

y_test_temp = yTest
Encoder = preprocessing.OneHotEncoder()
yTest = Encoder.fit_transform(yTest).toarray()
yTest = np.asarray(yTest, dtype=np.int32)
yTrain = Encoder.fit_transform(yTrain).toarray()
yTrain = np.asarray(yTrain, dtype=np.int32)

number = xTrain.shape[1]
batchSize = 128
epochs = 20         # 循环周期
numClasses = 10     # 分10类

model = Sequential()
# 1
model.add(Conv1D(filters=16, kernel_size=64, padding='same', input_shape=(xTrain.shape[1], 1)))
model.add(Conv1D(filters=32, kernel_size=32, padding='same'))
model.add(MaxPooling1D(60))

# 2
model.add(Conv1D(filters=64, kernel_size=3, padding='same'))
model.add(MaxPooling1D(2))

# 3
model.add(Conv1D(filters=256, kernel_size=3, padding='same'))
model.add(MaxPooling1D(5))

# 4
model.add(Conv1D(filters=512, kernel_size=3, padding='same'))
model.add(MaxPooling1D(1))

# 5
model.add(Dense(256))
model.add(Dropout(0.3))

# 6
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(16))

# 7
model.add(Dense(numClasses, activation='softmax'))
# model.summary()
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
tb_cb = TensorBoard(log_dir='logs')     # 查看训练情况
H = model.fit(x=xTrain, y=yTrain, batch_size=batchSize, epochs=epochs, verbose=1, shuffle=True, callbacks=[tb_cb])      # 开始模型训练

# 加载模型
# json_file = open(r'CNN-LSTM model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# model.load_weights('CNN-LSTM model.h5')
# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

score = model.evaluate(x=xTest, y=yTest, verbose=0)     # 训练结果
print("测试集上的损失：", score[0])
print("测试集上的准确度：", score[1])

font = {'family': 'Times New Roman'}
plt.rc('font', **font)

N = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.title("Training loss and Accuracy(Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

# 模型转换为json
# model_json = model.to_json()
# with open(r'CNN-LSTM model.json', 'w') as json_file:
#     json_file.write(model_json)
# model.save_weights('CNN-LSTM model.h5')

# testPredict = model.predict(test)
# print("预测结果：\n")
# for i in range(testPredict.shape[0]):
#     print(np.argmax(testPredict[i]))

yPredict = model.predict(xTest)         # 画出ROC曲线
plotAUCforMulticlass.plot_multiclass_AUC(numClasses, yTest, yPredict, 'CNN-LSTM')
plotPRCurveforMulticlass.plot_multiclass_PR(numClasses, yTest, yPredict, 'CNN-LSTM')

# 混淆矩阵
yPredict = np.argmax(yPredict, axis=-1)
# print(yPredict)
cm = confusion_matrix(y_test_temp, yPredict)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()