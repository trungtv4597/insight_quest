""""""

############# LOGGING
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

############# IMPORT

import os
from typing import List, Dict, Optional
import mimetypes
import pandas as pd

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.extractors import SummaryExtractor
from llama_index.core.indices import load_index_from_storage
from llama_index.core.prompts import RichPromptTemplate
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.program.evaporate.df import DFRowsProgram
from llama_index.core.query_engine import RetrieverQueryEngine
from pydantic import BaseModel

# lOCAL
from s3_operator import get_data_from_s3, upload_data_to_s3, download_data_from_s3

############# GLOBAL CONFIGURATIONS

from dotenv import load_dotenv

class Config:
    """
    Centralizes configuration, ensures required variables are set and reduces global namespace pollution.
    """
    def __init__(self):
        load_dotenv()

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

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
Settings.llm = OpenAI(
    api_key=config.OPENAI_API_KEY,
    model=config.OPENAI_FM,
    temperature=0.8
)
Settings.embed_model = OpenAIEmbedding(
    api_key=config.OPENAI_API_KEY,
    model=config.OPENAI_EMBED
)

# LlamaIndex - Tokenizer
import tiktoken
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
tokenizer = tiktoken.encoding_for_model(config.OPENAI_FM).encode
token_counter = TokenCountingHandler(tokenizer=tokenizer, verbose=True)
Settings.callback_manager = CallbackManager([token_counter])

############# CODE

def sync_cache_local_and_s3(config: Config) -> None:
    """
    It ensures that cache's and S3's version are similar.
    """
    if os.path.isdir(config.LOCAL_CACHE_PATH):
        os.makedirs(config.LOCAL_CACHE_PATH)
        return
    
    for file_name in os.listdir(config.LOCAL_CACHE_PATH):
        file_path = os.path.join(config.LOCAL_CACHE_PATH, file_name)
        s3_key = f"{config.S3_LLAMAINDEX_STORAGE_PATH}{file_name}"
                
        if os.path.isfile(file_path):
            # os.remove(file_path) # [BACKLOG] Add logic to compare file metadata (e.g. size, modification time) before syncing
            download_data_from_s3(s3_key=s3_key, file_path=file_path)
            logger.info(f"Sync <{file_name}>: <Done>")
        else:
            logger.error(f"Sync <{file_name}>: <Fail>")

def update_cache_to_s3(config: Config) -> None:
    """
    Re-upload all cache files (modified) from local to S3
    """
    if os.path.isdir(config.LOCAL_CACHE_PATH):
        for file_name in os.listdir(config.LOCAL_CACHE_PATH):
            file_path = os.path.join(config.LOCAL_CACHE_PATH, file_name)
            s3_key=config.S3_LLAMAINDEX_STORAGE_PATH+file_name
            # [BACKLOG] Check if upload is needed (e.g., compare hashes)
            upload_data_to_s3(file_path=file_path, s3_key=s3_key)
            # logger.info(f"file_path: <{file_name}> \n s3_key: <{S3_LLAMAINDEX_STORAGE+file_name}>")    

def load_documents(s3_data: Optional[Dict[str, bytes]]) -> List[Document]:
    """
    Convert S3 data into LlamaIndex Documents, mimicking SimpleDirectoryReader behviors.
    Args:
        - s3_data: Output from get_data_from_s3, expected as dict {key: bytes} (single or multiple files)
    Returns:
        - List of LlamaIndex Document objects
    """
    if s3_data is None:
        logger.error(f"No data recived from: <{s3_data}>")
        return []

    documents = []
    supported_mimetypes = {"text/plain", "application/pdf", "application/json"}
    
    for key, file_bytes in s3_data.items():

        # Skip if key is a folder
        if key.endswith("/"):
            continue

        # Determine file pe
        mime_type, _ = mimetypes.guess_type(key)
        if mime_type not in supported_mimetypes:
            logger.warning(f"Unsupported file type <{mime_type}> for <{key}>")
            continue

        try:
            # Convert bytes to text
            text = file_bytes.decode("utf-8", errors="ignore")
            doc = Document(
                text=text,
                doc_id=key,
                metadata={
                    "file_path": key,
                    "mime_type": mime_type or "unknown"
                }
            )
            documents.append(doc)
        except Exception as e:
            logger.error(f"Error processing file <{key}>: {e}")

    if not documents:
        logger.error("No valid dcuments created from s3_data")

    return documents

def load_and_cache_documents(config: Config) -> List[Document]:
    s3_data = get_data_from_s3(s3_key=config.S3_DOCUMENT_STORAGE_PATH, mode="all")
    # documents = load_documents(s3_data)
    # logger.info(f"Load documents after getting raw fomat from s3: <Done> | Len: {len(documents)}")
    return load_documents(s3_data)

def process_into_nodes(documents: List[Document], config: Config) -> List:
    # cache_data = get_data_from_s3(s3_key=S3_INGESTION_CACHE_FILE)
    # ingestion_cache = IngestionCache(cached_hashes=cache_data)
    ingestion_cache = IngestionCache.from_persist_path(config.LOCAL_INGESTION_CACHE)
    # logger.info(f"Get cache_ingestion from S3: <Done>")

    pipeline = IngestionPipeline(
        transformations=[
            TokenTextSplitter(chunk_size=1024, chunk_overlap=20),
            SummaryExtractor(summaries=["self"]),
            Settings.embed_model
        ],
        cache=ingestion_cache
    )

    nodes = pipeline.run(documents=documents)
    pipeline.cache.persist(config.LOCAL_INGESTION_CACHE)
    # logger.info(f"Nodes Parsing: <Done> | Len: {len(nodes)}")
    return nodes

def ingestion(config: Config) -> None:
    """
    This function loads <documents> (raw data) and chunks into <nodes>:
    Workflow:
        1. Load data from S3
            * Documents
            * Ingestion Cache
        2. Chunk data into nodes
            * Split documents
            * Extract metadata
        3. Store <nodes> config-files
            * Local cache
            * Upload to S3
    """
    documents = load_and_cache_documents(config)
    nodes = process_into_nodes(documents, config)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)
    storage_context.persist(persist_dir=config.LOCAL_CACHE_PATH)

def indexing(config: Config) -> None:
    """
    Index nodes for retrieve
    """
    # Load nodes config
    storage_context = StorageContext.from_defaults(persist_dir=config.LOCAL_CACHE_PATH)
    nodes = list(storage_context.docstore.docs.values())

    # Indexing
    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
    vector_index.set_index_id("vector")
    logger.info(f"Indexing: <Done>")
    
    # Save indecies config 
    storage_context.persist(persist_dir=config.LOCAL_CACHE_PATH)

def querying(config: Config) -> str:
    """
    Enable query engine to answer the prompts
    """
    # Load indices
    storage_context = StorageContext.from_defaults(persist_dir=config.LOCAL_CACHE_PATH)
    vector_index = load_index_from_storage(storage_context, index_id="vector")

    # Prompting
    temple = RichPromptTemplate(
        """
        Task: {{task_str}}
        ---
        Role: {{role_str}}
        ---
        Context: {{context_str}}
        ---
        Difficulty Level: {{difficulty_str}}
        ---
        Output Structure: {{output_str}}
        ---
        Avoidance Quiz: {{avoidance_str}}
        """
    )
    prompt_str = temple.format(
        task_str="Need to create a boolean quiz question based ONLY on provided documents",
        role_str="Act as a studying assistant who is helping me to reinforce my memories on specific knowledge through asking related question and providing rationle if I answer incorrect",
        context_str="I always take notes on all the information from my research journeys or daily, but usually forgot the insight after a period of time. I bebeilve it's a good way to improve the siutation by a doing examination regularly",
        difficulty_str="straightforward but mixing a little tricky",
        output_str=(
            "Return a JSON object with the following structure: "
            "{\"question\": \"<quiz question>\", \"answer\": <boolean>, \"rationale\": \"<explanation>\"}. "
            "The 'answer' field must be a boolean value (true or false, lowercase, no quotes, no punctuation). "
            "Example: {\"question\": \"Is the sky blue?\", \"answer\": true, \"rationale\": \"The sky appears blue due to Rayleigh scattering.\"}"
        )
    )

    # Query Engine
    query_engine = vector_index.as_query_engine()
    try:
        response = query_engine.query(prompt_str)
        # logger.info(f"LLM Response: <Done>")

        # logger.info("Token Usage:")
        # logger.info(f"\tEmbedding Tokens: {token_counter.total_embedding_token_count}")
        # logger.info(f"\tLLM Prompt Tokens (Input): {token_counter.prompt_llm_token_count}")
        # logger.info(f"\tLLM Completion Tokens (Output): {token_counter.completion_llm_token_count}")
        # logger.info(f"\tTotal LLM Token Count: {token_counter.total_llm_token_count}")

        return str(response)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return ""

def construct_llm_response(llm_response: str, config: Config) -> None:
    """
    Construct LLM's response to store a csv file.
    1. Set up a DataFrame to structure the quiz questions
    2. Extract dataframe-format in LLM's response 
    [BACKLOG] 3. Insert a new records to db
    """
    if not llm_response:
        logger.error("Empty LLM response")
        return
    
    df = pd.DataFrame({
        "question": pd.Series(dtype="str"),
        "answer": pd.Series(dtype="boolean"),
        "rationale": pd.Series(dtype="str")
    })

    # Construct Output schema
    class OutputSchema(BaseModel):
        question: str
        answer: bool
        rationale: str

    try:
        # Init DF extractor
        df_rows_program = DFRowsProgram.from_defaults(
            pydantic_program_cls=OpenAIPydanticProgram,
            pydantic_program_kwargs={"output_cls": OutputSchema},
            df=df
        )
        result_obj = df_rows_program(input_str=llm_response)
        df_quiz = result_obj.to_df(existing_df=df)
        # logger.info(f"DataFram Extraction: <Done>")
        # logger.info(f"DataFrame Contents:\n{df_quiz}")
        # logger.info(f"DataFrame dtypes:\n{df_quiz.dtypes}")
        df_quiz.to_csv(config.LOCAL_QUIZ_CACHE, index=False)
    except Exception as e:
        logger.error(f"Failed to contruct DataFram: {e}")

def create_quiz() -> None:
    """
    Entry point
    """
    # sync_cache_local_and_s3()
    ingestion(config)
    indexing(config)
    # update_cache_to_s3()
    llm_response = querying(config)
    # logger.info(f"LLM Response: \n{llm_response}")
    construct_llm_response(llm_response=llm_response, config=config)

if __name__ == "__main__":
    create_quiz()

    

