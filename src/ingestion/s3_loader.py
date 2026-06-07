"""
s3_loader.py

Responsibility: Connect to AWS S3 and download documents
to a local temporary directory for processing.

Why S3? Because production ML pipelines don't read from
local filesystems — data lives in object storage. This
makes the pipeline portable: run it on your laptop, on
an EC2 instance, or in a Lambda function without changing
the data layer.
"""

import boto3
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env file so os.getenv() can read your AWS credentials
load_dotenv()

# Set up logging — professional projects log what they're doing
# so you can debug problems without adding print statements everywhere
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(name)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)


class S3DocumentLoader:
    """
    Downloads documents from S3 to a local directory.
    
    Why a class? Because we need to maintain state (the S3 client,
    bucket name) across multiple method calls. A function would
    need these passed every time.
    """
    
    def __init__(self):
        """Initialize the S3 client using credentials from .env"""
        
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        self.region = os.getenv('AWS_REGION', 'us-east-1')
        
        if not self.bucket_name:
            raise ValueError(
                "S3_BUCKET_NAME not found in environment variables. "
                "Did you fill in your .env file?"
            )
        
        # Create the boto3 S3 client
        # boto3 automatically reads AWS_ACCESS_KEY_ID and 
        # AWS_SECRET_ACCESS_KEY from environment variables
        self.s3_client = boto3.client(
            's3',
            region_name=self.region
        )
        
        logger.info(f"S3DocumentLoader initialized — bucket: {self.bucket_name}")
    
    def list_documents(self, prefix: str = 'documents/') -> list[str]:
        """
        List all document keys in the S3 bucket under a given prefix.
        
        Args:
            prefix: The S3 "folder" to look in. Default is 'documents/'
            
        Returns:
            List of S3 keys (file paths inside the bucket)
        """
        logger.info(f"Listing documents in s3://{self.bucket_name}/{prefix}")
        
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            logger.warning("No documents found in S3 bucket.")
            return []
        
        # Filter out the prefix itself (S3 sometimes returns the 
        # folder as an object with 0 bytes)
        keys = [
            obj['Key'] 
            for obj in response['Contents']
            if obj['Size'] > 0
        ]
        
        logger.info(f"Found {len(keys)} documents")
        return keys
    
    def download_document(
        self, 
        s3_key: str, 
        local_dir: str = 'data/raw'
    ) -> Path:
        """
        Download a single document from S3 to local disk.
        
        Args:
            s3_key:    The full S3 key (e.g. 'documents/paper.pdf')
            local_dir: Where to save the file locally
            
        Returns:
            Path to the downloaded file
        """
        # Create local directory if it doesn't exist
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        
        # Extract just the filename from the S3 key
        # 'documents/attention_is_all_you_need.pdf' → 'attention_is_all_you_need.pdf'
        filename = Path(s3_key).name
        local_file_path = local_path / filename
        
        # Don't re-download if we already have it
        # This saves time and AWS data transfer costs
        if local_file_path.exists():
            logger.info(f"Already exists locally, skipping: {filename}")
            return local_file_path
        
        logger.info(f"Downloading: {s3_key} → {local_file_path}")
        
        self.s3_client.download_file(
            Bucket=self.bucket_name,
            Key=s3_key,
            Filename=str(local_file_path)
        )
        
        return local_file_path
    
    def download_all_documents(
        self,
        prefix: str = 'documents/',
        local_dir: str = 'data/raw'
    ) -> list[Path]:
        """
        Download all documents from S3 to local disk.
        
        Args:
            prefix:    S3 prefix (folder) to download from
            local_dir: Local directory to save files
            
        Returns:
            List of local file paths that were downloaded
        """
        keys = self.list_documents(prefix)
        
        if not keys:
            return []
        
        local_paths = []
        
        for key in keys:
            local_path = self.download_document(key, local_dir)
            local_paths.append(local_path)
        
        logger.info(f"Downloaded {len(local_paths)} documents to {local_dir}/")
        return local_paths


# --- Quick test ---
# When you run this file directly (python src/ingestion/s3_loader.py),
# this block executes. When you import it, it doesn't.
# This is the standard Python pattern for testable modules.
if __name__ == '__main__':
    loader = S3DocumentLoader()
    
    # List what's in S3
    keys = loader.list_documents()
    print(f"\nDocuments in S3:")
    for key in keys:
        print(f"  {key}")
    
    # Download the first one as a test
    if keys:
        downloaded_path = loader.download_document(keys[0])
        print(f"\nTest download successful: {downloaded_path}")
        print(f"File size: {downloaded_path.stat().st_size / 1024:.1f} KB")