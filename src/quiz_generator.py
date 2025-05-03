"""
Querying
"""

########## LOGGING

import logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

########## GLOBAL SETTINGS

import os
from dotenv import load_dotenv

class Config:
    """
    Centralizes configuration, ensures required variables are set and reduces global namespace pollution.
    """
    def __init__(self):
        load_dotenv()

        # Document Storage Map
        self.LOCAL_DOCUMENT_MAP = os.getenv("LOCAL_DOCUMENT_MAP")

        # LlamaIndex - OpenAI
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.OPENAI_FM = os.getenv("OPENAI_FM")
        self.OPENAI_EMBED = os.getenv("OPENAI_EMBED")

        # LlamaIndex-Store @ S3
        self.S3_DOCUMENT_STORAGE_PATH = os.getenv("S3_DOCUMENT_STORAGE_PATH")
        self.S3_INGESTION_CACHE_FILE = os.getenv("S3_INGESTION_CACHE_FILE")
        self.S3_LLAMAINDEX_STORAGE_PATH = os.getenv("S3_LLAMAINDEX_STORAGE_PATH")

        # Local Cache
        self.LOCAL_CACHE_PATH = os.getenv("LOCAL_CACHE_PATH")
        self.LOCAL_INGESTION_CACHE = os.getenv("LOCAL_INGESTION_CACHE")
        self.LOCAL_QUIZ_CACHE = os.getenv("LOCAL_QUIZ_CACHE")

    def _get_env(self, key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Environment variable {key} is not set")

config = Config()

########## LLAMAINDEX SETTINGS

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
Settings.llm = OpenAI(
    api_key=config.OPENAI_API_KEY,
    model=config.OPENAI_FM
)
Settings.embed_model = OpenAIEmbedding(
    api_key=config.OPENAI_API_KEY,
    model=config.OPENAI_EMBED
)

########## CODE

from prompting import (
    prompt_temple,
    task,
    context,
    role
) 

import pandas as pd
import random
from typing import Dict
import json
from tqdm import tqdm

from llama_index.core import StorageContext
from llama_index.core.indices import load_index_from_storage
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.program.evaporate.df import DFRowsProgram


def metadata_filter() -> Dict:
    """
    """
    df = pd.read_csv(r"C:\Users\Dell\OneDrive\Documents\documents\map_of_content.csv")

    # Ensure required columns exist
    required_columns = ['subject', 'subcategory']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV must contain 'subject', 'subcategory' columns")
    
    # Get a random subject
    subjects = df['subject'].unique()
    selected_subject = random.choice(subjects)
    
    # Filter for subcategories under the selected subject
    subcategories = df[df['subject'] == selected_subject]['subcategory'].unique()
    selected_subcategory = random.choice(subcategories)

    # Difficulty Level
    selected_difficulty = random.choice(["junior engineer", "senior engineer"])
    
    return {
        'subject': selected_subject,
        'subcategory': selected_subcategory,
        'difficulty': selected_difficulty,
    }

def existing_response_filter(config: Config, subject: str, subcategory: str) -> str:
    """
    Filters and retrieves unique questions from a CSV file based on subject, subcategory.

    Args:
        subject (str): The main subject area to filter questions.
        subcategory (str): The subcategory within the subject.

    Returns:
        str: A comma-separated string of unique questions matching the criteria.
             Returns empty string if no matches found or if an error occurs.

    Raises:
        FileNotFoundError: If the CSV file cannot be located at the specified path.
        pd.errors.EmptyDataError: If the CSV file is empty.
        KeyError: If required columns ('subject', 'subcategory', 'question') are missing.
        Exception: For other unexpected errors during execution.

    Example:
        >>> existing_response_filter("Math", "Algebra", "Linear Equations")
        "Solve x + 3 = 7,What is a linear equation?"

    Notes:
        - The function assumes the CSV file is located at:
          "C:\\Users\\Dell\\OneDrive\\Documents\\documents\\map_of_content.csv"
        - The CSV must contain columns: 'subject', 'subcategory', 'question'
        - Duplicate questions are removed using set conversion
    """
    try:
        # Attempt to read the CSV file
        df = pd.read_csv(config.LOCAL_QUIZ_CACHE)

        # Verify required columns exist
        required_columns = {'subject', 'subcategory', 'question'}
        if not required_columns.issubset(df.columns):
            raise KeyError(f"CSV missing required columns: {required_columns - set(df.columns)}")

        # Filter questions based on criteria and convert to list
        existing_questions = df[
            (df['subject'] == subject) & 
            (df['subcategory'] == subcategory)
        ]["question"].to_list()

        # Return unique questions joined by commas
        return ",".join(set(existing_questions)) if existing_questions else ""

    except FileNotFoundError:
        print(f"Error: CSV file not found at {config.LOCAL_QUIZ_CACHE}")
        return ""
    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty")
        return ""
    except KeyError as e:
        print(f"Error: {str(e)}")
        return ""
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return ""
    
def construct_llm_response(config: Config, llm_response: str, subject, subcategory, difficulty) -> None:
    """"""
    if not llm_response:
        logger.error("Empty LLM response")
        return
    
    df = pd.DataFrame({
        "question": pd.Series(dtype="str"),
        "answer": pd.Series(dtype="boolean"),
        "rationale": pd.Series(dtype="str")
    })
        
    # Init DF extractor
    df_rows_program = DFRowsProgram.from_defaults(
        pydantic_program_cls=OpenAIPydanticProgram,
        df=df
    )
    result_obj = df_rows_program(input_str=llm_response)
    df = result_obj.to_df(existing_df=df)

    df["subject"] = subject
    df["subcategory"] = subcategory
    df["difficulty"] = difficulty

    # Save
    df.to_csv(config.LOCAL_QUIZ_CACHE, mode="a", index=False, header=False)
    logger.info("Successfully save a new quiz to cache")

def generate_quiz(config: Config, num_quizzes: int) -> None:
    """
    """
    # Load Indexes
    storage_context = StorageContext.from_defaults(persist_dir=config.LOCAL_CACHE_PATH)
    vector_index = load_index_from_storage(storage_context, index_id="vector")
    # Init QueryEngine
    query_engine = vector_index.as_query_engine()

    for _ in tqdm(range(num_quizzes), desc="Generating quizzes ..."):
        # Metadata
        metadata = metadata_filter()
        subject = metadata["subject"]
        subcategory = metadata["subcategory"]
        difficulty = metadata["difficulty"]

        # Extract existing responses
        avoiding_questions = existing_response_filter(config, subject=subject, subcategory=subcategory)

        prompt = prompt_temple.format(
            context=context,
            task=task,
            role=role,
            difficulty_level=difficulty,
            subject=subject,
            subcategory=subcategory,
            avoid=avoiding_questions,
            output=(
                "Return a JSON object with the following structure: "
                "{\"question\": \"<quiz question>\", \"answer\": <boolean>, \"rationale\": \"<explanation>\"}. "
                "The 'answer' field must be a boolean value (True or False, no punctuation). "
                "Example: {\"question\": \"Is the sky blue?\", \"answer\": True, \"rationale\": \"The sky appears blue due to Rayleigh scattering.\"}"
            )
        )
        
        try:
            response = query_engine.query(prompt)
            logger.info(f"Query done")
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return ""
        
        # Contruct Outcomes
        construct_llm_response(config, llm_response=response, subject=subject, subcategory=subcategory, difficulty=difficulty)

if __name__ == "__main__":
    num_quizzes = 100
    generate_quiz(config, num_quizzes)
