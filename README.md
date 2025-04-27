# insight_quest: Quiz Generator and Telegram Bot

`insight_quest` is a Python application that generates boolean quiz questions from documents stored in AWS S3 using [LlamaIndex]() and delivers them as interactive quizzes via a Telegram bot by [python-telegram-bot](). The project aims to help users reinforce knowledge by creating and scheduling quizzes based on their notes or research documents (or personal data).

## Features
* `Quiz Generation`: 
** `Ingest` documnets from S3, `index` them, and `generate` boolean quiz questions with *LlamaIndex and OpenAI's* Foundation Models.
** `Cache` processed data locally and syncs with *AWS-S3* for efficient operation. 
* `Telegram Integration`: 
** Sends quizzes to a Telegram chat in `quiz mode`.
** `Schedules` one-time and recurring quiz delivery via Telegram commands.

## Project Structure

```mermaid
A[insight_quest] --> A1[src]
A --> A2[cache]
A --> .env
A --> requirements.txt
A1 --> B1[app.py]
A1 --> B2[quiz_generator.py]
A1 --> B3[s3_operator.py]
```

## Installation

### Configure Environment Variables

* Create a `.env` file in the project root with the following:
```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_FM=gpt-4
OPENAI_EMBED=text-embedding-ada-002
S3_DOCUMENT_STORAGE_PATH=s3://your-bucket/documents/
S3_INGESTION_CACHE_FILE=s3://your-bucket/cache/ingestion.json
S3_LLAMAINDEX_STORAGE_PATH=s3://your-bucket/cache/
LOCAL_CACHE_PATH=cache/
LOCAL_INGESTION_CACHE=cache/ingestion.json
LOCAL_QUIZ_CACHE=cache/quiz.csv
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
TOTAL_VOTER_COUNT=1
```

### Dependencies

*Key libraries used:

** `llama-index`: For document processing and quiz generation
** `python-telegram-bot`: For Telegram bot functionality
** `pandas`: For quiz data handling
** `boto3`: For AWS S3 interactions (via `s3_operator`)
** `python-dotenv`: For environment variable management

* Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

1. **Run the Application**:
   ```bash
   python app.py
   ```

2. **Interact with the Telegram Bot**:
   - Use the `/quiz` command to schedule quizzes:
     ```text
     /quiz h 1 24  # Send a quiz every 1 hour for 24 hours
     ```
   - The bot sends a boolean quiz question with a rationale for incorrect answers.
   - The poll closes automatically after the specified number of participants (set via `TOTAL_VOTER_COUNT`).
