import tensorflow as tf
import pandas as pd

inputfile1='Datareport.xlsx'
inputfile2=''
data = pd.read_excel(inputfile1,index='Date',sheet_name=0)
data_train_feature = data.loc[0:140,['数据大小','数据类型','数据格式','市场调研','产业经济','经营管理','智能投顾','车辆信息','商品信息','海关进出口'
          ,'知识产权','企业综合','电子商务']]


mean = data_train_feature.mean(axis=0)
std = data_train_feature.std(axis=0)

data_input=pd.read_excel(inputfile2,index='Date',sheet_name=0)

data_predict=data_input.loc[:,['数据大小','数据类型','数据格式','市场调研','产业经济','经营管理','智能投顾','车辆信息','商品信息','海关进出口'
          ,'知识产权','企业综合','电子商务']]

data_predict=(data_predict-mean)/std

model=tf.keras.models.load_model('Datareport.h5')

result=model.predict()