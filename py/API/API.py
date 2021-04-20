import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

def Root_Mean_Squared_Logarithmic_Error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
inputfile='API.xlsx'

data = pd.read_excel(inputfile,index='Date',sheet_name=0)
data_train_feature = data.loc[0:780,['智能识别','产业经济','电子商务','商品信息','短信API','企业综合','企业工商','银行卡核验','车辆信息','智能风控','天气查询','经营管理'
          ,'身份核验','知识产权','app应用','快递查询','舆情监测','IP地址','手机号验证','海关进出口','智能客服','1分钱','司法','银行卡信息',
          '交通违章','资质备案','行政监管','位置地图','招投标','公告文书','投融资','黑名单','风控','手机号码归属','反欺诈','星座运势','手机号码状态',
           '航空航班','新闻资讯','环境质量','尾号限行','行驶驾驶','税务信息','信用评估','基站','自然灾害','万年历','手机在网时长','企业图谱','智能营销',
           '油价查询','彩票信息','公路铁路','京东E卡','市场调研','借贷','智能支付','区号查询','用户画像','股票汇率','视频会员']]
data_train_price=data.loc[0:780,['价格']]

mean = data_train_feature.mean(axis=0)
std = data_train_feature.std(axis=0)
data_train_feature = (data_train_feature - mean)/std #数据标准化
x_train = data_train_feature
y_train = data_train_price




def model_build():
   model=tf.keras.Sequential()
   model.add(tf.keras.layers.Dense(15,activation="tanh",input_shape=(61,)))
   model.add(tf.keras.layers.Dense(15,activation="relu"))
   model.add(tf.keras.layers.Dense(10,activation="sigmoid"))
   model.add(tf.keras.layers.Dense(1))
   model.summary()
   model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.3),loss=Root_Mean_Squared_Logarithmic_Error,metrics=["mae"])
   return model


k = 4
num_val_samples = len(data_train_feature) // k
num_epochs = 100
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
model.save('API.h5')
