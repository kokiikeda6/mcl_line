# mcl_line
## 概要
1次元の数直線上を動くロボットにmclを実装したものになります.

## 実行例
[mcl_line.py](https://github.com/kokiikeda6/mcl_line/blob/main/mcl/mcl_line.py)を実行すると以下のような画面が現れ, ロボットが右向きに動き始めます. \
ロボットが動くと数直線上にパーティクルが広がり, ランドマークを観測すると狭まります. \
(赤色の円: ロボット, 青色の矢印: パーティクル, 黄色の星: ランドマーク)
<img src="resource/mcl_demo.gif">

## 実行手順
1) ターミナル上でインストールしたいディレクトリに移動
2) インストール
```
git clone https://github.com/kokiikeda6/mcl_line.git
```
3) 実行
```
cd mcl
./mcl_line.py
```
## パラメータ
|パラメータ名|説明|
|:---|:---|
|time_span|シミュレート時間 [s]|
|time_interval|時間間隔 [s]|
|num_particles|パーティクルの数|
|initial_pose|ロボットの初期位置(x座標)|
|velocity|ロボットの並進速度 [m/s]|
|nn|直進1[m]で生じる道のりのばらつきの標準偏差|

## 動作確認済み環境
* Python 3.8.10
* Ubuntu 20.04

## 参考
* 上田隆一『詳解 確率ロボティクス -Pythonによる基礎アルゴリズムの実装-』講談社, 2019年.