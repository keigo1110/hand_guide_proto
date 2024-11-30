# Hand Guidance System

カメラを使用して手の動きを追跡し、インタラクティブなガイドラインに沿って指先の動きを誘導するシステムです。MediaPipeを使用して手の検出を行い、指定されたパスに沿って指先を動かすことで、ハンドトラッキングの練習や評価が可能です。機会があればMR環境にアップデート予定です。

## 機能

- リアルタイムの手指追跡
- セグメント化されたガイドラインの表示
- 進捗状況の視覚的フィードバック
- 完了時の自動検出

## 必要条件

- Python 3.7以上
- OpenCV (`cv2`)
- MediaPipe
- NumPy

## インストール

1. リポジトリをクローンします：
```bash
git clone https://github.com/keigo1110/hand_guide_proto.git
cd hand_guide_proto
```

2. 必要なパッケージをインストールします：
```bash
pip install opencv-python mediapipe numpy
```

## 使用方法

1. プログラムを実行します：
```bash
python hand_guide_dis.py
```

2. カメラが起動し、画面上にガイドラインが表示されます。
3. 手を画面に向けて、人差し指の先端でガイドラインをなぞります。
4. 各セグメントは以下の色で表示されます：
   - 緑：未到達のセグメント
   - 黄：現在のターゲットセグメント
   - 赤：完了したセグメント
   - 青：全セグメント完了時

5. 'q'キーを押すとプログラムが終了します。

## カスタマイズ

`HandGuidanceSystem`クラスのインスタンス化時に以下のパラメータを調整できます：

```python
guidance_system = HandGuidanceSystem(
    num_segments=20,      # セグメントの数
    line_start=(100, 100),  # 開始位置（x, y）
    line_end=(400, 400)     # 終了位置（x, y）
)
```

## 技術的な詳細

- MediaPipeを使用して手の検出とランドマークの追跡を行います
- 指先の位置は人差し指の先端（`INDEX_FINGER_TIP`）を使用
- セグメントへの近接判定は閾値ベースで行われます（デフォルト：10ピクセル）