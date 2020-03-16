# 数据清洗，替换城市别名
import requests
import json
import time
import pandas as pd

data = pd.read_csv('province_data.csv')
alias = pd.read_csv('alias_city.csv')

# 别名替换
def alias_replace(x):
    if x in alias['alias'].values.tolist():
        temp = alias[alias['alias']==x]
        result = temp['city'].values[0]
    else:
        result = x

        if result[-1:] == '区':
            if result[-3:] == '开发区' or result[-2:] == '新区':
                result = x
            else:
                result = x[:-1]
        if result[-1:] == '县':
            result = x[:-1]
        if result[-1:] == '市':
            result = x[:-1]
    return result

province_data = data[data['province']==data['city']]
city_data = data.drop(province_data.index)
province_data.to_csv('province_data1.csv', index=False)
city_data['city'] = city_data['city'].apply(alias_replace)
city_data.to_csv('city_data1.csv', index=False)
current_city = city_data.drop_duplicates(['province', 'city'], keep = 'first')
#print(result)
#print(alias['alias'])
#print(current_city)
current_city.to_csv('current_city1.csv', index=False)