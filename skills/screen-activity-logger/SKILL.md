---
name: screen-activity-logger
description: 画面録画（MP4）をデフォルトはローカルで作業ログ化する。「画面録画を作業ログに」「録画からログを作って」「会議録画をログ化」「この録画で何をしていたか」「screen recording to worklog」等の依頼で使用。OCR+VLM+ASRの3コンテキストを統合し、タイムスタンプ・滞留時間・アプリ別滞在時間つきの日本語ログを生成する。
---

# screen-activity-logger — 画面録画→作業ログ（ローカル既定）

録画済みMP4を入力に、①画面の文字（OCR）②何をしているかの説明（VLM）③発話（ASR）を
統合したタイムスタンプ付き作業ログ（Markdown＋JSONL）を生成する。
ローカルVLMを選ぶ限りクラウドAPIには送らない。Gemini / Anthropicを選ぶ場合は、
フレームが外部送信されるため、許可・保持設定・顧客契約を確認してから実行する。

リポジトリ: https://github.com/iloli-source/screen-activity-logger

## 前提チェック（実行前に毎回確認）

```bash
which uv ffmpeg
# ローカルVLMを使う場合だけ ollama --version も確認する
```

足りないものがあればOS別に案内する:

| ツール | macOS | Windows |
|---|---|---|
| uv | `brew install uv` | `winget install astral-sh.uv` |
| ffmpeg | `brew install ffmpeg` | `winget install Gyan.FFmpeg` |
| Ollama（ローカルVLM時） | https://ollama.com/download | https://ollama.com/download/windows |

## セットアップ（初回のみ）

```bash
git clone https://github.com/iloli-source/screen-activity-logger.git  # (public化後に有効)
cd screen-activity-logger
uv venv -p 3.12 .venv
uv sync --locked --extra asr                       # lock済み依存を同期
ollama pull qwen3-vl:8b                              # 約6GB、初回のみ
```

セットアップ済みかは `.venv/bin/screen-activity-logger --help`（Windows: `uv run screen-activity-logger --help`）で確認できる。

## モード判定（ユーザーの言葉から選ぶ）

- 「会議」「Meet」「Zoom」「打ち合わせ」の録画 → **`--mode meeting`**（話者切替の無駄打ちを抑制）
- 操作デモ・チュートリアル・作業画面の録画 → **既定（screencast）**（操作ステップを落とさない）
- どちらか不明なら**ユーザーに確認する**（間違えると取りこぼし or 冗長化する）

## 実行

```bash
# 単発
.venv/bin/screen-activity-logger 録画.mp4 -o out/

# 会議録画
.venv/bin/screen-activity-logger 会議.mp4 --mode meeting -o out/

# 複数本は必ず1コマンドでバッチ（並列起動はメモリ破綻するため禁止）
.venv/bin/screen-activity-logger 会議1.mp4 会議2.mp4 --mode meeting -o out/
```

### クラウドVLMを使う場合（明示オプトイン）

APIキーは環境変数に置き、`--model`、`--vlm-api-key-env`、
`--allow-external-vlm`を必ず明示する。キーの実値をCLI引数には渡さない。

```powershell
$env:GEMINI_API_KEY = "キーの実値"
uv run screen-activity-logger 録画.mp4 --no-asr `
  --vlm-provider gemini --model <gemini-model> `
  --vlm-api-key-env GEMINI_API_KEY --allow-external-vlm -o out/
```

Anthropicを使う場合は `--vlm-provider anthropic` と `ANTHROPIC_API_KEY` を指定する。
クラウド利用時は機密録画・顧客録画を送信してよいかを先に確認する。

VLMサーバーやAPIキーがまだない場合は、`--no-vlm` を明示してOCR/ASR基線を実行できる。
これは外部送信なしで抽出・出力を検証するためのモードで、VLMによる画面理解とは別物である。

処理時間の目安: 5分の録画で15分前後（VLM推論が支配項）。長い場合は
バックグラウンド実行にしてユーザーに待ち時間を伝えること。

## 結果の案内

- `out/worklog.md` — 人間用。見出し `## 00:05:00〜00:12:30（7分30秒） — Excel — ファイル名` に
  滞留時間、`👁`=注視箇所、`🗣️`=発話、末尾に**アプリ別滞在時間**テーブル
- `out/worklog.jsonl` — 機械用。`duration_seconds` でアプリ別集計可、`ocr` に一次情報（生OCR）を全保持

完了時は worklog.md の見出し数行とアプリ別滞在時間を要約してユーザーに見せる。

## トラブルシュート

| 症状 | 対処 |
|---|---|
| `VLM呼び出し失敗: ReadTimeout` | 自動で1回リトライ済み。頻発するなら `--vlm-timeout 450` で延長、他の重い処理を止める |
| 音声トラックなし | 自動でASRスキップ（正常動作） |
| Windowsで遅い | faster-whisperはCUDA自動判別。NVIDIA GPUなら `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` |
| `ASRバックエンド未インストール` | エラーメッセージ内の導入コマンドを実行、または `--no-asr` |
| `ASRバックエンドは見つかりましたが読み込めません` | Python/OSに合うwheelを再構築、または `--no-asr` |

詳細オプションは README.md の「使い方」を参照（このスキルと重複管理しない）。
