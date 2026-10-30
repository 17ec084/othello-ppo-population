# GPT-2型の小型モデル20体によるオセロ強化学習

[Qiitaの実験記事](https://qiita.com/hiratatomotaka/items/f44497a4321fd3b7ae0a)の再現用ソースです。学習コードは記事の4,085 step時点の実装を収録しています。

## どの実験を再現するか

20体から学習者を1体選び、他の19体それぞれと128局（先後64局ずつ）、合計2,432局を対戦させます。勝率が55%を厳密に超えれば追加更新をせず別の個体へ交代し、未達ならその対局からPPOで学習します。相手の重み・価値関数は固定します。報酬は勝ち+1・負け-1・引き分け0。方策の部分凍結・共有学習・異端学習・エントロピー報酬はありません。

GPT-2型のdecoder-only Transformer（4層、幅96、4ヘッド、方策1,084,576パラメータ）をランダム初期化します。8行を各1トークンにし、読み出しトークンを追加した長さ9の入力です。PPOの価値層は97パラメータ。学習・固定器評価とも温度1で確率サンプリングします。

記事の初期シードは20261028です。完全な設定は [`data/experiment-config.json`](data/experiment-config.json)、実行時のバージョンは [`data/environment.json`](data/environment.json) にあります。

## 1. 環境を作る

元の環境はWindows、Python 3.13.1、NumPy 2.2.1、PyTorch 2.9.1+cu130、RTX 3070 Laptop GPUです。学習本体と以下のPython起動スクリプトはOS固有の起動処理に依存しませんが、今回の配布確認はWindows上で行っています。

Pythonの仮想環境を作り、使用するPythonで次を実行してください。

```shell
python -m venv .venv
```

Windows PowerShellの場合:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOSの場合:

```shell
source .venv/bin/activate
```

```shell
python -m pip install -r requirements.txt
```

GPU版PyTorchは、使用GPU・ドライバーに合わせて[PyTorch公式のインストール手順](https://pytorch.org/get-started/locally/)を使用してください。元環境と同じCUDA 13.0向けPyTorch 2.9.1を選ぶ場合は、[公式の過去バージョン用手順](https://pytorch.org/get-started/previous-versions/#v291)に従い次のコマンドです。

```shell
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. 先に小さく動作確認する

```shell
python reproduce.py --smoke
```

CPU・3体・小型モデル・2 stepの確認です。記事の学習条件とは異なります。出力先は `runs/smoke/`。既存の出力先に上書きして新規学習することはありません。

## 3. 記事と同じ条件で最初から学習する

```shell
python reproduce.py --updates 4085 --out runs/reproduction-4085
```

20体・2,432局/step・同じシードとハイパーパラメータで開始します。記事では終了上限なしで走らせ、4,085 stepで手動停止しました。上のコマンドは、その時点までを上限にしたものです。

上限なしで続けるなら:

```shell
python reproduce.py --updates 0 --out runs/reproduction-unlimited
```

GPUが利用できる環境では自動でCUDAを使います。明示する場合は `--device cuda` または `--device cpu` を追加します。CPUで本番条件を回すと大幅に時間がかかります。

元の実行は評価・保存・容量待ち込みで約35時間でした。少なくとも4 GiBの空き容量を用意し、実行中は2 GiBの余裕を確保します。RAM/VRAM不足によるページファイル増加など、チェックポイント以外のディスク使用量にも余裕を持たせてください。Windowsのページファイル設定は変更しません。

## 4. 進捗画面を見る

別のターミナルで、同じフォルダから起動します。

```shell
python monitor.py --port 8766
```

ブラウザで http://127.0.0.1:8766/ を開くと、個体別成績・全個体平均・学習担当・交代履歴・代表対局を確認できます。`latest_run.txt` がこのフォルダ内で最後に起動した実験を指します。

## 5. 保存して停止し、続きを再開する

別のターミナルで:

```shell
python request_stop.py
```

実行中のstepと保存が終わるまで待ち、画面が「保存して停止済み」になったことを確認してください。ファイルで確認する場合は `progress.json` の `status` と `phase` が両方 `stopped` になるまで待ちます。

```shell
python reproduce.py --resume runs/reproduction-4085
```

総step数の上限を増やす場合（`--updates` は追加step数ではありません）:

```shell
python reproduce.py --resume runs/reproduction-4085 --updates 5000
```

`initial.pt` と `latest.pt.gz` の両方を保持してください。重み・Adam・価値関数・担当者・乱数状態を復元します。実行環境が変われば浮動小数点演算やCUDAの非決定性により、同じシードでも曲線が完全一致する保証はありません。

## 6. 記事の図だけを再描画する

学習を何日も回さなくても、記事で使った全4,086時点の評価履歴から図を作れます。

```shell
python -m pip install -r requirements-plot.txt
python plot_results.py
```

`figures/` に全個体平均・AI 0・AI 12・AI 11のPNGを出力します。日本語フォントが必要です。WindowsではMeiryo、LinuxなどではNoto Sans CJK JP等を利用してください。

自分の実験の図を出す場合:

```shell
python plot_results.py --run runs/reproduction-4085
```

同梱データは評価結果と担当者履歴です。学習済み重み・全対局棋譜は含みません。学習の追試と、記録済みの図の再描画は別の操作です。

## 7. 検証

```shell
python -m unittest -v test_winrate test_ppo test_individual test_reward_comparison test_runtime
```

先後の対戦数、55%の境界、選ばれていない個体の固定、達成時の更新省略、停止・再開と連続実行の一致などを検証します。Node.jsがあればグラフの回帰テストも実行できます。

```shell
node test_individual_chart.cjs
```

## 主なファイル

| ファイル | 役割 |
|---|---|
| `individual_train.py` | 20体の対戦・評価・学習・保存 |
| `individual_control.py` | 55%超で学習者を交代する制御 |
| `model.py` | 行トークンとGPT-2型方策モデル |
| `othello.py` | オセロのルール・行トークン化・位置評価表 |
| `ppo.py` | PPOと価値関数・GAE |
| `compact_checkpoint.py` | 圧縮保存・復元 |
| `reproduce.py` | 記事の条件で起動・再開する配布用ラッパー |
| `plot_results.py` | 記録済みデータまたは自分の実験から図を作成 |
| `data/` | 記事の評価曲線、設定、環境、集計結果 |

`train.py` や `model.py` などには、それ以前に試した実験方式も残っています。過去方式の関数を共用しているため、元ファイルを保って収録しました。記事の条件で起動する入口は **`reproduce.py`** です。元のPowerShell起動スクリプトも併せて収録しています。

元コードのファイル一覧とSHA-256は `SOURCE_MANIFEST.json` に記録しています。学習本体は元実験からそのままコピーし、配布用の説明・起動ラッパー・描画の入口を追加しています。
