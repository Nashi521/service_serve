import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K

def Root_Mean_Squared_Logarithmic_Error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

inputfile='Datareport.xlsx'

data = pd.read_excel(inputfile,index='Date',sheet_name=0)

data_train_feature = data.loc[0:140,['数据大小','数据类型','数据格式','市场调研','产业经济','经营管理','智能投顾','车辆信息','商品信息','海关进出口'
          ,'知识产权','企业综合','电子商务']]
data_train_price= data.loc[0:140,['价格']]


mean = data_train_feature.mean(axis=0)
std = data_train_feature.std(axis=0)
data_train_feature = (data_train_feature - mean)/std #数据标准化
x_train = data_train_feature
y_train = data_train_price



def model_build():
   model=tf.keras.Sequential()
   model.add(tf.keras.layers.Dense(15,activation="tanh",input_shape=(13,)))
   model.add(tf.keras.layers.Dense(15,activation="relu"))
   model.add(tf.keras.layers.Dense(10,activation="relu"))
   model.add(tf.keras.layers.Dense(1))
   model.summary()
   model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),loss=Root_Mean_Squared_Logarithmic_Error,metrics=["mae"])
   return model


k = 4
num_val_samples = len(data_train_feature) // k
num_epochs = 500
all_scores = []
for i in range(k):
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k（准备验证数据：第 k 个分区的数据）
    val_data = data_train_feature[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = data_train_price[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions（准备训练数据：其他所有分区的数据）
    partial_train_data = np.concatenate(
        [data_train_feature[:i * num_val_samples],
         data_train_feature[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [data_train_price[:i * num_val_samples],
         data_train_price[(i + 1) * num_val_samples:]],
        axis=0)

    # Build the Keras model (already compiled)（构建 Keras 模型（已编译））
    model = model_build()
    # Train the model (in silent mode, verbose=0)（训练模型（静默模式，）
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=32)
    # Evaluate the model on the validation data（在验证数据上评估模型）
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    print(all_scores)
    print(np.mean(all_scores))

K.clear_session()
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k（准备验证数据：第 k 个分区的数据）
    val_data = data_train_feature[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = data_train_price[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions（准备训练数据：其他所有分区的数据）
    partial_train_data = np.concatenate(
        [data_train_feature[:i * num_val_samples],
         data_train_feature[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [data_train_price[:i * num_val_samples],
         data_train_price[(i + 1) * num_val_samples:]],
        axis=0)

    # Build the Keras model (already compiled)（构建 Keras 模型（已编译））
    model = model_build()
    # Train the model (in silent mode, verbose=0)（训练模型（静默模式，verbose=0））
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=32, verbose=0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

model=model_build()
model.fit(x_train,y_train,epochs=100,batch_size=32)
model.save('Datareport.h5')