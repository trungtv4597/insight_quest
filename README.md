# insight_quest: Quiz Generator and Telegram Bot

`insight_quest` is a Python application that generates boolean quiz questions from documents stored in AWS S3 using [LlamaIndex](https://docs.llamaindex.ai/en/stable/) and delivers them as interactive quizzes via a Telegram bot by [python-telegram-bot](https://docs.python-telegram-bot.org/en/v21.10/examples.html). The project aims to help users reinforce knowledge by creating and scheduling quizzes based on their notes or research documents (or personal data).

## Features
* `Quiz Generation`: 
** `Ingest` documnets from S3, `index` them, and `generate` boolean quiz questions with *LlamaIndex and OpenAI's* Foundation Models.
** `Cache` processed data locally and syncs with *AWS-S3* for efficient operation. 
* `Telegram Integration`: 
** Sends quizzes to a Telegram chat in `quiz mode`.
** `Schedules` one-time and recurring quiz delivery via Telegram commands.

## Project Structure

```
insight_quest/ 
├── src
│   ├── app.py # Telegram bot implementation
│   ├── quiz_generator.py # Logic for quiz generation using LlamaIndex
│   └── s3_operator.py # S3 utility functions
├── cache # Local cache for LlamaIndex and quiz data
├── .env # Environment variables (not tracked)
├── requirements.txt # Python dependencies
└── README.md
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

* Key libraries used:

** `llama-index`: For document processing and quiz generation
** `python-telegram-bot`: For Telegram bot functionality
** `pandas`: For quiz data handling
** `boto3`: For AWS S3 interactions (via `s3_operator`)
** `python-dotenv`: For environment variable management

* Install all dependencies with:
```bash
pip install -r requirements.txt
```

# Deployment Guide for the program on AWS EC2

## Prerequisites
* AWS `EC2` instance: A `running` instance with a `public IP`.
* `PuTTY`: installed on your local machine, configured with a `.ppk` key for SSH access to EC2.
* `AWS Credentials`: Configured for `S3` access (via IAM role)
* `GitHub Repository`

## Workflow Overview
1. *Commit and push* code changes to `GitHub` from the local machine.
2. *Connect* to `EC2` instance using PuTTY.
3. Set up the EC2 environment (*clone repository, install dependencies*).
4. *Create* and configure the `.env` file on EC2.
5. *Configure* AWS credentials for `S3 access`.
6. Test the application.
7. Run the bot in the *background* using `nohup`.
8. Update the application with new changes.

## Step-by-Step Deployment

### Step 1: Commit and Push Changes to GitHub

1. Navigate to Project Directory: On the local machine, open `Git Bash` and got to the project repo.
```bash
cd path/to/your/project
```

2. Stage Files: Add all necessary files.
```bash
git add app.py requirements.txt quiz_generator.py
```

3. Commit Changes: Record the changes with a descriptive message.
```bash
git commit -m "Add Telegram quiz bot and dependencies"
```

4. Ensure `.gitignore`: Prevent sensitive files like `.evn` from being pushed.
```bash
echo ".env" >> .gitignore
git add .gitignore
git commit -m "Add .gitignore for .env"
```

5. Push to Github: Upload changes to the GitHub repo
```bash
git push
```

### Step 2: Connect to EC2 with PuTTY

1. Open PuTTY: Launch PuTTY on the local machine.
2. Configure Session:
   * In the `Session` category:
      * Host Name: Enter the EC2 instance's public IP or DNS.
   * In the `Connection` cateogry:
      * Connection > SSH > Auth: upload `.ppk` key file
   * Click `Open` to connect
3. Log In:
   * Enter the username: 
      * Amazon Linux: `ec2-user`
      * Ubuntu: `ubuntu`

### Step 3: Set Up the EC2 Environment
1. Update the Instance
```bash
sudo yum update -y  # Amazon Linux 2
# OR
sudo apt update && sudo apt upgrade -y  # Ubuntu
```

2. Install Git
```bash
sudo yum install git -y  # Amazon Linux 2
# OR
sudo apt install git -y  # Ubuntu
```

3. Install Python 3 and pip
```bash
sudo yum install python3 python3-pip -y  # Amazon Linux 2
# OR
sudo apt install python3 python3-pip -y  # Ubuntu
```

4. Clone the GitHub Repository
```bash
cd ~ # Navigate to the home directory
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

5. Create Virtual Environment
```bash
source venv/bin/activate
```

6. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### Step 4: Create and Configure the .env File

1. Create the .env file in the project directory
```bash
nano .env
```

2. Add Environment Variable
```text
TELEGRAM_BOT_TOKEN=your_bot_token_from_botfather
TELEGRAM_CHAT_ID=your_chat_id
TOTAL_VOTER_COUNT=1
POLLING_INTERVAL=1.0
S3_QUIZ_FILE=your/s3/path/to/quiz.csv
PEAK_START_HOUR=9
PEAK_END_HOUR=21
PEAK_INTERVAL_MINUTES=30
OFF_PEAK_INTERVAL_MINUTES=120
```

3. Secure the file
```bash
chmod 600 .env # restrict access
ls -l .env # verify
```

### Step 5: Configure AWS Credentials for S3 Access

### Step 6: Run the Bot in the Background

1. Stop any running instances
```bash
ps aux | grep python3 # Check for exising processes
kill -9 PID # replace with the actual ID
```

2. Run with `nohup`
```bash
nohup python3 src/main.py > bot.log 2>&1 &
```

3. Verify
```bash
ps aux | grep python3
cat bot.log # View initial logs
tail -f bot.log # Monitor logs in real-time
```

4. Close PuTTY

### Step 7: Update the Application

1. Push Changes Locally
```bash
git add .
git commit -m "Update quiz bot"
git push origin main
```

2. Connect to EC2: use PuTTY to log in

3. Stop the current running session

4. Pull Updates

5. Restart the program


## Tools Introduction

* `Git`: A version control system for tracking code changes and collaboarting via repositories.
* `Github`: A platform for hosting Git repositories, enabling code sharing and version control.
* `PuTTY`: A free SSH client for Windows, used to connect to and manage remote servers like EC2 via a terminal interface.
* `SSH (Secure Shell)`: A protocol for secure remote access to servers, using a private key (*.ppk*) for authentication.
* `.ppk` key file is converted from the EC2 `.pem` key using `PuTTYgen`.
* `yum/apt`: Package managers for Amazon Linux to install software.
* `.env` file: a text file storing environment variables (e.g., API tokens, S3 paths) used by the bot via the `python-dotenv` package.
* `nohup`: A linux command to run processes immune to session termination, ensuring the bot continues after PuTTY disconnection.
* `AWS IAM Role`: A secure way to grant EC2 access to AWS resources without storing credentials.
