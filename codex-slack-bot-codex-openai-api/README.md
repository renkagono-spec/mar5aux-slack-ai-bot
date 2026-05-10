# Slack AI Search Bot MVP

Slack内の投稿を保存し、`@Company AI 質問` で関連メッセージを検索してOpenAI APIで回答するMVPです。完成形はローカル常駐ではなく、Slack App + 公開Webサーバー + Postgres + OpenAI APIで動きます。

## できること

- Slack Events APIで投稿を取り込む
- botが参加しているチャンネルの投稿をPostgresへ保存する
- メッセージ本文をOpenAI embeddingsで検索用に保存する
- `@bot 今週ロゴ周りで何が決まった？` のような質問にSlackスレッドで返答する
- 回答にSlack permalinkの根拠を含める
- 検索ヒットの同一スレッドと前後の投稿もContextに含める
- 既存ログを `scripts/backfill.py` で取り込む

## 質問する場所

質問は同じSlackワークスペース内で、botを招待したチャンネルからできます。

`SEARCH_SCOPE=workspace` の場合は、botが参加していてDBに保存済みのチャンネルを横断検索します。private channelの内容をpublic channelへ出したくない場合は、専用のprivateチャンネル、たとえば `#company-ai` を作り、`ALLOWED_ANSWER_CHANNEL_IDS` にそのチャンネルIDを入れてください。

## 必要なもの

- Slack App
- Slack Bot Token: `xoxb-...`
- Slack Signing Secret
- OpenAI API key
- Postgresの `DATABASE_URL`
- Render / Railway / Cloud Run などの公開URL

## Slack App設定

1. Slack APIで新しいAppを作成
2. `slack-app-manifest.example.json` をベースにManifestを設定
3. `request_url` を `https://YOUR_PUBLIC_HOST/slack/events` に変更
4. Appをワークスペースへインストール
5. 対象チャンネルにbotを招待

必要な主なEventは以下です。

- `app_mention`
- `message.channels`
- `message.groups`
- `message.im`
- `message.mpim`

## 環境変数

`.env.example` を参照してください。本番では最低限これが必要です。

```bash
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://user:password@host:5432/slack_ai
SEARCH_SCOPE=workspace
```

## デプロイ

Docker対応済みです。

```bash
docker build -t slack-ai-search .
docker run --env-file .env -p 8080:8080 slack-ai-search
```

公開URLの `/healthz` が返ればサーバー側は起動しています。Slack AppのRequest URLには `/slack/events` を指定してください。

Render Blueprintで作る場合は `render.yaml` も入っています。GitHubリポジトリをRenderに接続し、BlueprintまたはWeb Serviceとして作成したあと、Environment Variablesに秘密情報を入れてください。

## 既存ログの取り込み

Events APIは基本的に「これから流れる投稿」を受け取ります。過去ログも検索したい場合は、対象チャンネルIDを指定してバックフィルします。

```bash
python scripts/backfill.py C12345678 C23456789 --max-messages 1000
```

private channelはbotを招待してから実行してください。

## MVP後に足すもの

- pgvectorでDB内ベクトル検索
- 添付ファイル本文の抽出
- 日次/週次サマリー
- 未対応タスク抽出
- 案件別タイムライン
- 権限別の回答制御
