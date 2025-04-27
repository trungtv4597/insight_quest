""""""

########## LOGGING

import logging
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger.addHandler(handler)

########## IMPORT

# lOCAL
from quiz_generator import create_quiz

# Standard Library
import os
import pandas as pd
from datetime import datetime
from typing import Dict

# Thrid-Party
from telegram import (
    Update,
    Poll
)
from telegram.ext import (
    ContextTypes,
    Application,
    CommandHandler,
    PollHandler
)

########## GLOBAL SETTING

from dotenv import load_dotenv

class Config:
    """
    Centralizes configuration, ensures required variables are set and reduces global namespace pollution.
    """
    def __init__(self):
        load_dotenv()

        # Telegram
        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
        self.TOTAL_VOTER_COUNT = os.getenv("TOTAL_VOTER_COUNT")

        # Cache
        self.LOCAL_QUIZ_CACHE = "cache/quiz.csv"
    
    def _get_env(self, key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Missing {key}")
        return value
    
config = Config()

########## CODE

class InsightQuest:
    """
    Clase to orchestrate the Insight Quest quiz game via Telegram.
    """
    def __init__(self,):
        self.config = Config()
        self.create_quiz()
        self.quiz = self.load_quiz()

    def create_quiz(self):
        create_quiz()

    def load_quiz(self) -> Dict[str, any]:
        """
        Load quiz data from CSV file.
        """
        try:
            df = pd.read_csv(self.config.LOCAL_QUIZ_CACHE)
            if not df.empty:
                return df.iloc[0].to_dict()
            else:
                logger.error("Quiz data is empty.")
                return {}
            
        except Exception as e:
            logger.error(f"Error loading quiz data: {str(e)}")
            return {}

    async def send_quiz(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Send a predefined quiz
        """
        if not self.quiz:
            logger.error("No quiz data available")
            return
        
        options = ["True", "False"]
        question = self.quiz.get("question", "No question available")
        answer = str(self.quiz.get("answer", True))
        rationale = self.quiz.get("rationale", "No rationale available")
        correct_option_id = 1 if answer else 0
        # logger.info(f"Quiz: \n\tQuestion: <{question}> \n\tAnswer: <{answer}> type: <{type(answer)}> \n\tCorrect_Option_ID: <{correct_option_id}>")
    
        job = context.job
        message = await context.bot.send_poll(
            chat_id=job.chat_id,
            question=question,
            options=options,
            type=Poll.QUIZ,
            allows_multiple_answers=False,
            correct_option_id=correct_option_id,
            explanation=rationale
        )

        # Save some info about the quiz in bot_data for later use.
        payload = {
            message.poll.id: {
                "chat_id": job.chat_id,
                "message_id": message.message_id
            }
        }
        context.bot_data.update(payload)

    async def receive_quiz_answer(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Close quiz after be answered (reaching to the limit participants = 1)
        """
        poll = update.poll
        if poll.is_closed:
            return
        if poll.total_voter_count == int(self.config.TOTAL_VOTER_COUNT):
            try:
                quiz_data = context.bot_data[poll.id]
                await context.bot.stop_poll(quiz_data["chat_id"], quiz_data["message_id"])
                logger.info(f"Poll {poll.id} closed")
            except KeyError:
                logger.warning(f"No quiz data found for poll ID <{poll.id}>")
            except Exception as e:
                logger.error(f"Error stopping poll: {str(e)}")

    def remove_job_if_exists(self, name: str, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Remove job with given name. Returns whether job was removed."""
        current_jobs = context.job_queue.get_jobs_by_name(name)
        if not current_jobs:
            return False
        for job in current_jobs:
            job.schedule_removal()
        return True

    async def set_timer(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Add a job to the queue to send a quiz."""
        chat_id = update.effective_message.chat_id
        try:
            due = float(context.args[0])
            if due < 0:
                await update.effective_message.reply_text("Sorry, we cannot go back in time!")
                return

            job_removed = self.remove_job_if_exists(str(chat_id), context)
            context.job_queue.run_once(self.send_quiz, due, chat_id=chat_id, name=str(chat_id), data=due)

            text = "Timer successfully set for quiz!"
            if job_removed:
                text += " Old timer was removed."
            await update.effective_message.reply_text(text)

        except (IndexError, ValueError):
            await update.effective_message.reply_text("Usage: /set <seconds>")   

    async def set_scheduler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Add a repeating job to the queue to send quizzes"""
        chat_id = update.effective_message.chat_id
        try:
            # Extract interval, and last from command arguments.  
            scale, interval, last = context.args
            interval, last = float(interval), float(last)

            if interval <= 0 or last <= 1:
                await update.message.reply_text("Interval must be positive and last > 1 second")
                return
            
            multiplier = {
                "m": 60,
                "h": 60*60,
                None: 1 # Default case
            }.get(scale, 1)

            interval *= multiplier
            last *= multiplier
            
            self.remove_job_if_exists(str(chat_id), context)
            context.job_queue.run_repeating(
                self.send_quiz,
                interval=interval, 
                last=last,
                chat_id=chat_id,
                name=str(chat_id),
                data=interval
            )

            await update.effective_message.reply_text(f"Repeating quiz schedule set! Quizzes every {interval}{scale}, until {last}{scale} from now")

        except (IndexError, ValueError):
            await update.effective_message.reply_text(
                "Usage: /schedule <scale: (h)our/(m)inute/(s)econd> <interval: float> <last: float>"
            )


def main() -> None:
    """"""
    try:
        # create_quiz()
        application = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
        insight_quest = InsightQuest()

        # Handler
        application.add_handler(CommandHandler("quiz", insight_quest.set_timer))
        application.add_handler(PollHandler(insight_quest.receive_quiz_answer))

        # Start the bot
        application.run_polling(poll_interval=10, allowed_updates=Update.ALL_TYPES)
    
    except Exception as e:
        logger.error(f"Application failed: {e}")

if __name__ == "__main__":
    main()