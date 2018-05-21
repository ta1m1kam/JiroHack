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
- 実行コマンド
```
$ curl -X POST -F image=@<img/to/path.jpg> 'http://localhost:5000/predict'
```
