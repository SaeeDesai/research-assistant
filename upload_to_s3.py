"""
One-time utility script to upload raw documents to S3.
Run this from the project root: python upload_to_s3.py
"""

import boto3
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def upload_documents_to_s3():
    """Upload all PDFs from data/raw/ to S3 bucket."""
    
    # Create S3 client using credentials from .env
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION')
    )
    
    bucket_name = os.getenv('S3_BUCKET_NAME')
    raw_data_path = Path('data/raw')
    
    # Find all PDFs in the raw data folder
    pdf_files = list(raw_data_path.glob('*.pdf'))
    
    if not pdf_files:
        print("No PDF files found in data/raw/")
        print("Please download some papers from arXiv and place them there.")
        return
    
    print(f"Found {len(pdf_files)} PDF files. Uploading to s3://{bucket_name}/documents/")
    print("-" * 50)
    
    uploaded = 0
    failed = 0
    
    for pdf_path in pdf_files:
        # S3 key = the "path" inside your bucket
        # We store all docs under a documents/ prefix
        s3_key = f"documents/{pdf_path.name}"
        
        try:
            s3_client.upload_file(
                Filename=str(pdf_path),
                Bucket=bucket_name,
                Key=s3_key
            )
            print(f"  ✓ Uploaded: {pdf_path.name}")
            uploaded += 1
            
        except Exception as e:
            print(f"  ✗ Failed:   {pdf_path.name} — {e}")
            failed += 1
    
    print("-" * 50)
    print(f"Done. {uploaded} uploaded, {failed} failed.")
    
    # Verify by listing what's in the bucket
    print(f"\nVerifying — listing files in s3://{bucket_name}/documents/")
    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix='documents/'
    )
    
    if 'Contents' in response:
        for obj in response['Contents']:
            size_kb = obj['Size'] / 1024
            print(f"  {obj['Key']} ({size_kb:.1f} KB)")
    else:
        print("  No files found — something went wrong.")

if __name__ == '__main__':
    upload_documents_to_s3()