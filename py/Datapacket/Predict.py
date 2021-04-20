import tensorflow as tf
import pandas as pd

inputfile1='Datapacket.xlsx'
inputfile2='../../excel/test1.xlsx'
data = pd.read_excel(inputfile1,index='Date',sheet_name=0)
data_train_feature = data.loc[0:1050,['数据大小','数据类型','数据格式','位置地图','新闻资讯','舆情监测','产业经济','市内交通','智能客服','企业工商'
          ,'企业图谱','自然灾害','智能识别','电子商务','企业综合','环境质量','投融资','商品信息','市场调研','公路铁路','经营管理','公告文书',
          'app应用','知识产权','天气查询','信用评估','资质备案']]

mean = data_train_feature.mean(axis=0)
std = data_train_feature.std(axis=0)

data_input=pd.read_excel(inputfile2,index='Date',sheet_name=0)

data_predict=data_input.loc[:,['数据大小','数据类型','数据格式','位置地图','新闻资讯','舆情监测','产业经济','市内交通','智能客服','企业工商'
          ,'企业图谱','自然灾害','智能识别','电子商务','企业综合','环境质量','投融资','商品信息','市场调研','公路铁路','经营管理','公告文书',
          'app应用','知识产权','天气查询','信用评估','资质备案']]

data_predict=(data_predict-mean)/std

model=tf.keras.models.load_model('Datapacket.h5')

result=model.predict()

print(result)

