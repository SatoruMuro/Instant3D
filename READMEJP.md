# Instant3D

**Instant3D** は、CTやMRIのDICOM 画像や NIfTI ファイルから臓器・組織を自動セグメンテーションし、3D STL モデルや解析用データを簡単に出力できるツールです。TotalSegmentator を内部で利用しており、医師・研究者が追加のプログラミングなしに 3D 再構築を実行できます。

![Instant3D Top Image](https://github.com/SatoruMuro/Instant3D/blob/main/files/Instant3D_image01.jpg)

---

## ダウンロードと配置

Instant3D の最新版は GitHub Releases から zip 形式で配布しています：
[Instant3D Releases ページ](https://github.com/SatoruMuro/Instant3D/releases/tag/Instant3Dv20250829)

1. 上記リンクから zip ファイルをダウンロードします。
2. zip を解凍し、フォルダごと `C:\\` ドライブ直下に配置してください（例：`C:\\Instant3D`）。
   → この配置でパスがシンプルになり、エラーを避けやすくなります。

---

## インストール（初回だけ）

### かんたん一括セットアップ（推奨・Windows 10/11）

1. **スタートメニュー → “Windows Terminal” または “コマンドプロンプト” を右クリック → 管理者として実行**
   （どちらでも可。Windows 11 では通常 “PowerShell” が開きます）

2. 以下を **丸ごとコピーして実行**（Python と TotalSegmentator をインストールし、モデルもダウンロードします）

```bat
:: === Instant3D セットアップ（管理者PowerShell/CMDで実行） ===
:: 1) Python を winget で入れる（既に入っていれば自動スキップ）
winget install -e --id Python.Python.3.12 -h || echo (Python 3.12: すでに入っている/手動インストール済み)

:: 2) pip を更新 & TotalSegmentator をインストール（CPU版）
python -m pip install --upgrade pip
pip install TotalSegmentator

:: 3) モデルをダウンロード（最初だけ数GB）
totalsegmentator --download_model

:: 4) 動作確認（ヘルプが表示されればOK）
totalsegmentator -h
echo.
echo === Setup complete! Close this window. ===
pause
```

⚠️ **winget が使えないPC** では、[Python 公式サイト](https://www.python.org/downloads/windows/) から **Python 3.12 (64bit)** をインストールしてください。インストール時に `Add Python to PATH` にチェックを入れ、その後 上記スクリプトの **2)〜4)** だけ実行します。

---

## 使い方（基本）

1. `Instant3D.exe` をダブルクリックで起動
2. **Input**: DICOM フォルダ または NIfTI ファイルを選択
3. ROI を入力し、**Add ROI** を押す（複数追加可）
4. **Run** を押す
5. 出力は自動で `<入力名>_Instant3D` フォルダに保存されます

<img src="https://github.com/SatoruMuro/Instant3D/blob/main/files/Instant3D_image02.png" width="50%">


---

## 対象構造物（ROI）一覧（タスク別）

タスクごとの **対象構造物の一覧（ROI 名）** を、以下のフォルダに **.txt** で公開しています：
**[https://github.com/SatoruMuro/Instant3D/tree/main/files/resources](https://github.com/SatoruMuro/Instant3D/tree/main/files/resources)**

* ファイル例：`roi_catalog_body.txt`, `roi_catalog_body_mr.txt`, `roi_catalog_abdominal_muscles.txt`, `roi_catalog_brain_structures.txt` など
* 各ファイルは **TotalSegmentator のタスク**に対応しています（例：`body`, `body_mr`, `abdominal_muscles` …）。
* **ROI 入力欄**には、各 txt 内に記載された名称をそのまま入力してください（例：`liver`, `kidney_right`, `femur_left` など）。途中まで入力すれば候補が予測表示されます。
* いくつかの特別サブタスク（例：`appendicular_bones`, `tissue_types`, `heartchambers_highres`, `face` など）は **Academic ライセンス**が必要です。必要に応じて本 README 末尾の「TotalSegmentator: 特別サブタスクの Academic ライセンス」を参照してください。

さらに詳しい対象構造物やタスクの情報については、**TotalSegmentator 公式リポジトリ**も参照してください：
[https://github.com/wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator)

---

## よくある質問（FAQ）

**Q1. `totalsegmentator` が見つからないと言われる**
A. ターミナルで以下を実行し、パスが通っているか確認してください。

```sh
totalsegmentator -h
```

表示されない場合は、以下のフォルダを PATH に追加します（例）：

```
C:\Users\<あなたのユーザー名>\AppData\Local\Programs\Python\Python312\Scripts\
```

追加後、ターミナルを閉じて開き直し、再度 `totalsegmentator -h` を試してください。

---

**Q2. モデルのダウンロードに時間がかかる／途中で止まる**
A. ネットワーク環境によっては 30 分以上かかることがあります。容量は数 GB 程度必要です。

---

**Q3. GPU は必要？**
A. GPU がなくても動作します（CPU 設定あり）。GPU があると高速化される場合があります。

---

**Q4. どこに出力されますか？**
A. 入力の親フォルダ直下に `<入力名>_Instant3D` が作成されます。

---

## アップデート／アンインストール

### TotalSegmentator を更新

```sh
pip install -U TotalSegmentator
```

### モデルを再取得／更新

```sh
totalsegmentator --download_model
```

### アンインストール

```sh
pip uninstall TotalSegmentator
```

（Python 自体を削除する場合は、Windows の「アプリと機能」からアンインストールしてください）

---

## トラブル時のチェックリスト

* `Instant3D.exe` は実行できるか？
* `totalsegmentator -h` がヘルプ表示されるか？（されない＝PATH 設定の問題）
* モデルのダウンロードは完了しているか？（初回は時間がかかる）
* 書き込み先（`<入力名>_Instant3D`）にアクセス権があるか？（ネットワークドライブ注意）

---

## TotalSegmentator: 特別サブタスクの Academic ライセンス

Instant3D で利用される TotalSegmentator の中には、**appendicular\_bones**, **tissue\_types**, **heartchambers\_highres**, **face** など、一部のサブタスクが **学術／非商用向けの制限付きライセンス** のもとで提供されています。これらのタスクを使用するには、ライセンスキーを取得・登録する必要があります。

### ライセンスキーの取得方法

1. 以下のページにアクセスして学術ライセンスキーを申請してください：
   [Academic License for TotalSegmentator 特別モデル](https://backend.totalsegmentator.com/license-academic/)
2. 所属機関や使用目的を入力して申請します。
3. 承認されると、メール等で **ライセンスキー** が発行されます（例：`aca_XXXXXXXXXX`）。

### ライセンスキーの登録方法

ターミナルまたはコマンドプロンプトで以下を実行してください：

```sh
totalseg_set_license -l aca_XXXXXXXXXX
```

登録が成功すれば、該当するサブタスク（例：`--task appendicular_bones`）が利用可能になります。

> 注：この Academic ライセンスは **非商用利用限定** です。商用で利用する場合は、別途商用ライセンスを取得してください。

---

## 参考資料

3D データ（STL など）の開き方については、以下の PDF にまとめています：
[3D Slicer での 3D データの開き方（日本語）](https://github.com/SatoruMuro/Instant3D/blob/main/files/HowToOpen3D%283Dslicer%29JP.pdf)

---

## 引用について

研究や論文で TotalSegmentator を利用する場合は、**公式リポジトリ**に記載された推奨引用形式を参照してください：
[https://github.com/wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator)

---
