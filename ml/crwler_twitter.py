# -*- coding:utf-8 -*-

import requests
import requests_oauthlib
import json
import math
import imgutil
import twikey

# image save path
directory = "./img"
imgutil.mkdir(directory)

url = "https://api.twitter.com/1.1/search/tweets.json"

# parameters
query = "ramen"
lang = "ja"
result_type="mixed" # 最新のツイートを取得
count = 100 # 1回あたりの最大取得ツイート数（最大100）
max_id = ''

total_count = 1000 # 取得画像の最大数
offset = math.floor(total_count/count) # ループ回数

# oauthの設定
consumer_key = twikey.twikey['consumer_key']
consumer_secret = twikey.twikey['consumer_secret']
access_token = twikey.twikey['access_token']
access_secret = twikey.twikey['access_secret']
oauth = requests_oauthlib.OAuth1(consumer_key,consumer_secret,access_token,access_secret)

for i in range(offset):
    params = {'q':query,'lang':lang,'result_type':result_type,'count':count,'max_id':max_id}
    r = requests.get(url=url,params=params,auth=oauth)
    print(max_id)
    json_data = r.json()
    for data in json_data['statuses']:
        # 最後のidを格納
        max_id = str(data['id']-1)
        if 'media' not in data['entities']:
            continue
        else:
            for media in data['entities']['media']:
                if media['type'] == 'photo':
                    image_url = media['media_url']
                    try:
                        imgutil.download_img(directory,image_url)
                    except Exception as e:
                        print("failed to download image at {}".format(image_url))
                        print(e)
                        continue
