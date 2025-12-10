import os
from dotenv import load_dotenv
import boto3
from botocore.config import Config

load_dotenv()

def get_env(key, default=None):
    return os.getenv(key, default)

def get_s3_client():
    """
    Get a boto3 S3 client configured for LocalStack.
    Uses s3_force_path_style=True for LocalStack compatibility.
    """
    endpoint_url = get_env('S3_ENDPOINT_URL', 'http://localhost:4566')
    aws_access_key_id = get_env('AWS_ACCESS_KEY_ID', 'test')
    aws_secret_access_key = get_env('AWS_SECRET_ACCESS_KEY', 'test')
    region_name = get_env('AWS_REGION', 'us-east-1')
    
    # Configure boto3 for LocalStack with path-style addressing
    config = Config(
        s3={
            'addressing_style': 'path'
        }
    )
    
    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
        config=config
    )
    
    return s3_client

def get_s3_resource():
    """
    Get a boto3 S3 resource configured for LocalStack.
    Uses s3_force_path_style=True for LocalStack compatibility.
    """
    endpoint_url = get_env('S3_ENDPOINT_URL', 'http://localhost:4566')
    aws_access_key_id = get_env('AWS_ACCESS_KEY_ID', 'test')
    aws_secret_access_key = get_env('AWS_SECRET_ACCESS_KEY', 'test')
    region_name = get_env('AWS_REGION', 'us-east-1')
    
    # Configure boto3 for LocalStack with path-style addressing
    config = Config(
        s3={
            'addressing_style': 'path'
        }
    )
    
    s3_resource = boto3.resource(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
        config=config
    )
    
    return s3_resource
