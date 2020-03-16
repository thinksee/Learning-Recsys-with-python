# 从丁香园接口获取整体数据
import requests
import json
import time
import pandas as pd

# 获取HTML文本
def get_html_text(url):
    try:
        res = requests.get(url,timeout = 30)
        res.raise_for_status()
        res.encoding = res.apparent_encoding
        return res.text
    except:
        return "Error"

# 将timestamp转换为日期类型
def timestamp_to_date(timestamp, format_string="%Y-%m-%d"):
    time_array = time.localtime(timestamp)
    str_date = time.strftime(format_string, time_array)
    return str_date

# 从row中得到数据
def get_data_from_row(row, province, city, updateTime):
    confirmedCount = row['confirmedCount']
    confirmedCount = row['confirmedCount']
    suspectedCount = row['suspectedCount']
    curedCount = row['curedCount']
    deadCount = row['deadCount']
    temp_dict = {'province': province, 'city': city, 'updateTime': updateTime, 'confirmedCount': confirmedCount, 'suspectedCount': suspectedCount, 'curedCount': curedCount, 'deadCount': deadCount}
    return temp_dict

# 返回某个省份下面所有城市的数据
def get_data_from_cities(results, province, updateTime):
    data = []
    for row in results:
        print(row)
        cityName = row['cityName']
        temp_dict = get_data_from_row(row, province, cityName, updateTime)
        data.append(temp_dict)
    return data        
    #df = pd.DataFrame(data)
    #clean_df = df.drop_duplicates(['province', 'city', 'updateTime'], keep = 'first')
    #return clean_df.values.tolist()

# 得到指定的省份数据
def get_data_from_province(province = '全国'):
    if province == '全国':
        page_url = "https://lab.isaaclin.cn/nCoV/api/overall?latest=0"
    else:
        page_url = 'https://lab.isaaclin.cn/nCoV/api/area?latest=0&province=' + province

    data = []
    text = get_html_text(page_url)
    results = json.loads(text)["results"]
    for row in results:
        if 'updateTime' in row:
            updateTime = timestamp_to_date(row['updateTime'] / 1000)
        else:
            updateTime = timestamp_to_date(row['modifyTime'] / 1000)
        temp_dict = get_data_from_row(row, province, province, updateTime)
        data.append(temp_dict)

        if 'cities' in row and len(row['cities']) > 0:
            result2 = row['cities']
            print(type(result2))
            print(result2)

            df = get_data_from_cities(result2, province, updateTime)
            data.extend(df)
            #print(df)
            
            #df = get_data(row, province, city=False)

    df = pd.DataFrame(data)
    print(df)
    clean_df = df.drop_duplicates(['province', 'city', 'updateTime'], keep = 'first')
    #return df
    return clean_df

def get_province_name():
    #获取Json
    page_url = "https://lab.isaaclin.cn/nCoV/api/provinceName"
    text = get_html_text(page_url)
    province_list = json.loads(text)["results"]
    return province_list

province_list = get_province_name()
# 得到全国的总统计数据
result = get_data_from_province()
# 得到每个省份的统计数据
for province in province_list:
    df = get_data_from_province(province)
    print(df)
    result = pd.concat([result, df])
result.to_csv('province_data.csv', index=False)
