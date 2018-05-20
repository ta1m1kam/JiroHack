# -*- coding:utf-8 -*-
import OpenSSL
import requests
import urllib
import hashlib
import sha3
import os

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 引数fをファイル名と拡張子（.は含まない）に分割する
def split_filename(f):
    split_name = os.path.splitext(f)
    file_name =split_name[0]
    extension = split_name[-1].replace(".","")
    return file_name,extension

def download_img(path,url):
    mkdir(path)
    _,extension  = split_filename(url)
    if extension.lower() in ('jpg','jpeg','gif','png','bmp'):
        encode_url = urllib.parse.unquote(url).encode('utf-8')
        hashed_name = hashlib.sha3_256(encode_url).hexdigest()
        full_path = os.path.join(path,hashed_name.decode('utf-8') + '.' + extension.lower())

        r = requests.get(url)
        if r.status_code == requests.codes.ok:
            with open(full_path,'wb') as f:
                f.write(r.content)
            print('saved image...{}'.format(url))
        else:
            print("HttpError:{0}  at{1}".format(r.status_code,image_url))
