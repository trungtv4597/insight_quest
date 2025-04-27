import boto3
from botocore.exceptions import ClientError
import os
import json
from io import BytesIO
import tempfile
import shutil

# Global Settings
from dotenv import load_dotenv
load_dotenv()

# AWS-S3
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
import boto3
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name="ap-southeast-1"
)

import logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def get_data_from_s3(s3_key, mode=None) -> dict:
    """
    Retrieve data from S3 using get_object.
    Returns:
        - If model=None: bytes of the single file, or None if error
        - If model="all": dict with {key: bytes} for all files, or None if error
    """
    
    try:
        if mode == "all":
            # List objects in the folder (prefix)
            response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=s3_key)
            if "Contents" not in response:
                logger.error(f"No files found in {S3_BUCKET_NAME}/{s3_key}")
                return None
            
            # Fetch all files
            files_data = {}
            for obj in response["Contents"]:
                key = obj["Key"]
                # Skip if key is a folder (ends with "/")
                if key.endswith("/"):
                    continue
                obj_response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=key)
                files_data[key] = obj_response["Body"].read()
            # logger.info(f"Successfully fetch all files from {S3_BUCKET_NAME}/{s3_key}: \nKey:\t{str(files_data.keys())}")
            return files_data
        
        else:
            # Fetch a single file
            response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
            return {s3_key: response["Body"].read()}
        
    except ClientError as e:
        logger.error(f"Error retrieving <{S3_BUCKET_NAME}/{s3_key}>: {e}")
        return None

def put_data_to_s3(data, s3_key):
    """
    Upload data directly to S3 without local storage (Put)
    """
    try:
        # Convert data to bytes if necessary
        if isinstance(data, str):
            data = data.encode("utf-8")
        elif isinstance(data, dict):
            data = json.dumps(data).encode("utf-8")
        elif not isinstance(data, (bytes, BytesIO)):
            raise ValueError("Data must be str, dict, bytes, or BytesIO")
        
        # Put to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=data
        )
        logger.info(f"Sucessfully uploaded <{s3_key}> to <{S3_BUCKET_NAME}>")
        return True
    
    except ClientError as e:
        logger.error(f"Error uploading <{s3_key}>: {e}")
        return False
        
def upload_data_to_s3(file_path, s3_key):
    """
    Upload a file from a local disk to S3 bucket.
    """
    try:
        s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)
        logger.info(f"Sucessfully uploaded <{file_path}> to <s3://{S3_BUCKET_NAME}/{s3_key}>")
        return True
    
    except ClientError as e:
        logger.error(f"Error uploading <{file_path}>: {e}")
        return False

def download_data_from_s3(s3_key, file_path):
    """
    Download a file from S3 bucket to a local directory.
    """       
    try:
        s3_client.download_file(S3_BUCKET_NAME, s3_key, file_path)
        logger.info(f"Sucessfully downloaded <s3://{S3_BUCKET_NAME}/{s3_key}> to {file_path}")
        return True
    
    except ClientError as e:
        logger.error(f"Error downloading <{s3_key}>: {e}")
        return False

    
if __name__ == "__main__":
    folder_name = "upload_data/"
    get_data_from_s3(s3_key=folder_name, mode="all")
