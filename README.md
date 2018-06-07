ジロリアンによる二郎のための二郎アプリ

## 開発環境構築
- Ruby 2.5.0
- RoR 5.1.5

```
$ git clone <This repository>
$ cd <This repository name>
$ bundle exec bundle install
```

### Twitter Omniauth の環境変数
- dotenv-railsを利用

`dotenv-rails`をgem install 
```
gem 'dotenv-rails'
```

`app/`以下に`.env`ファイルを作成し、TwitterのAPI_KEYとAPI_SECRET_KEYを記入する。

# API(flask)
`ML/model`にmodel設置（拡張子.h5py）

- ローカルでのアプリ連携用
```
$ sudo python main.py
```

- 実行コマンド（curlテスト用）portとIPを修正すること(localhost, port=5000)
```
$ curl -X POST -F image=@<img/to/path.jpg> 'http://localhost:5000/predict'

例
$ cd ML
$ python main.py
$ curl -X POST -F image=@'./rabit.jpg' 'http://localhost:5000/predict'
{"predictions":[{"label":"Angora","probability":0.9945027828216553},{"label":"hare","probability":0.004503111355006695},{"label":"wood_rabbit","probability":0.00083166389958933},{"label":"hamster","probability":8.828080899547786e-05},{"label":"Persian_cat","probability":3.835480674752034e-05}],"success":true}
```

### Railsサーバー起動
これはスマホ用のWebアプリケーションであるのでローカルでは以下のように起動すること
```
$ bundle exec rails s -b <LANのIPアドレス>
```
