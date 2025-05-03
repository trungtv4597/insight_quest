"""
"""

########## IMPORT

import logging
import os
from dotenv import load_dotenv
from typing import Dict, Optional
import pandas as pd
import io
import datetime

from telegram import Update, Poll
from telegram.ext import Application, ContextTypes, PollHandler

from s3_operator import get_data_from_s3

########## Logging setup

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

##########

class Config:
    def __init__(self):
        load_dotenv()
        # Telegram
        self.TELEGRAM_BOT_TOKEN = self._get_env("TELEGRAM_BOT_TOKEN")
        self.TELEGRAM_CHAT_ID = self._get_env("TELEGRAM_CHAT_ID")
        self.TOTAL_VOTER_COUNT = self._get_env("TOTAL_VOTER_COUNT", "1")
        self.POLLING_INTERVAL = float(self._get_env("POLLING_INTERVAL", "10"))
        # S3
        self.S3_QUIZ_FILE = self._get_env("S3_QUIZ_FILE")
        # Schedule
        self.PEAK_START_HOUR_UTC = int(self._get_env("PEAK_START_HOUR_UTC"))
        self.PEAK_END_HOUR_UTC = int(self._get_env("PEAK_END_HOUR_UTC"))
        self.PEAK_INTERVAL_MINUTES_UTC = int(self._get_env("PEAK_INTERVAL_MINUTES_UTC"))
        self.OFF_PEAK_INTERVAL_MINTUES_UTC = int(self._get_env("OFF_PEAK_INTERVAL_MINTUES_UTC"))

    def _get_env(self, key: str, default: str = None) -> str:
        value = os.getenv(key, default)
        if not value:
            raise ValueError(f"Missing {key}")
        return value

config = Config()
########## APP

class InsightQuest:
    def __init__(self):
        self.config = Config()

    def load_quiz(self) -> Optional[Dict[str, any]]:
        try:
            s3_key = config.S3_QUIZ_FILE
            data = get_data_from_s3(s3_key=s3_key)
            if not data or s3_key not in data:
                logger.error(f"Failed to retrieve quiz data from s3: {s3_key}")
                return None
            csv_data = data[s3_key]
            df = pd.read_csv(io.BytesIO(csv_data))
            if not df.empty:
                return df.sample(n=1).iloc[0].to_dict()
            else:
                logger.error("Quiz data is empty.")
                return None
        except Exception as e:
            logger.error(f"Error loading quiz data: {str(e)}")
            return None
        
    def get_next_quiz_time(self, current_time: datetime.datetime) -> datetime.datetime:
        """Calculate the delay in seconds until the next quiz based on the current time"""
        current_hour = current_time.hour
        if config.PEAK_START_HOUR_UTC <= current_hour < config.PEAK_END_HOUR_UTC:
            interval = config.PEAK_INTERVAL_MINUTES_UTC
        else:
            interval = config.OFF_PEAK_INTERVAL_MINTUES_UTC
        
        next_time = current_time + datetime.timedelta(minutes=interval)
        logger.info(f"The quiz will be sent at {next_time}")
        return next_time
    
    def remove_job_if_exists(self, name: str, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Remove job with given name. Returns whether job was removed"""
        current_jobs = context.job_queue.get_jobs_by_name(name=name)
        if not current_jobs:
            return False
        for job in current_jobs:
            job.schedule_removal()
        return True
    
    async def send_quiz(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        job = context.job
        quiz = self.load_quiz()
        if not quiz:
            logger.error("No quiz data available")
            await context.bot.send_message(chat_id=job.chat_id, text="Failed to load quiz. Please try again")
            return
        
        options = ["True", "False"]
        answer = bool(quiz.get("answer", True))
        correct_option_id = 0 if answer else 1
        question = quiz.get("question", "No question available")
        rationale = quiz.get("rationale", "No rationale available")
        if len(question) > 255:
            question = question[:252] + "..."
        if len(rationale) > 200:
            rationale = rationale[:197] + "..."

        message = await context.bot.send_poll(
            chat_id=job.chat_id,
            question=question,
            options=options,
            type=Poll.QUIZ,
            allows_multiple_answers=False,
            correct_option_id=correct_option_id,
            explanation=rationale
        )

        payload = {
            message.poll.id: {
                "chat_id": job.chat_id, 
                "message_id": message.message_id
                }
            }
        context.bot_data.update(payload)

        # Clear any existing quiz job
        self.remove_job_if_exists("next_quiz", context=context)

        # Schedule the next quiz
        next_time = self.get_next_quiz_time(current_time=datetime.datetime.now(datetime.UTC))
        context.job_queue.run_once(self.send_quiz, next_time, chat_id=job.chat_id, name="next_quiz")

    async def receive_answer(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        poll = update.poll
        if poll.is_closed:
            return
        if poll.total_voter_count == int(config.TOTAL_VOTER_COUNT):
            try:
                quiz_data = context.bot_data[poll.id]
                await context.bot.stop_poll(quiz_data["chat_id"], quiz_data["message_id"])
                logger.info(f"Poll {poll.id} is closed")
            except KeyError:
                logger.warning(f"No quiz data found for poll ID <{poll.id}>")
            except Exception as e:
                logger.error(f"Error stopping poll: {str(e)}")

def main() -> None:
    try:
        my_application = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
        my_bot = InsightQuest()

        # Schedule the first quiz
        current_time = datetime.datetime.now(datetime.UTC)
        first_quiz_time = my_bot.get_next_quiz_time(current_time)
        my_application.job_queue.run_once(
            my_bot.send_quiz,
            first_quiz_time,
            chat_id=config.TELEGRAM_CHAT_ID,
            name="next_quiz"
        )

        # Add poll handler
        my_application.add_handler(PollHandler(my_bot.receive_answer))

        # Start the bot
        my_application.run_polling(poll_interval=config.POLLING_INTERVAL, allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.error(f"Application failed: {e}")

if __name__ == "__main__":
    main()
    # current_time = datetime.datetime.now(datetime.UTC)
    # my_bot = InsightQuest()
    # next_time = my_bot.get_next_quiz_time(current_time=current_time)
    # print(f"Current UTC Time: {current_time} - Next Time: {next_time}")