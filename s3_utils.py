"""
AWS S3 Utilities for Simpson Pipeline
Handles file uploads to S3 bucket with error handling
"""

import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Optional


def get_s3_client():
    """
    Initialize and return S3 client using IAM role from EC2 instance
    
    When running on EC2 with an IAM role, boto3 automatically retrieves:
    - Credentials from the instance metadata service
    - Region from the instance metadata (if not specified)
    
    Returns:
        boto3.client: S3 client instance
    """
    # boto3 will automatically use:
    # 1. Credentials from EC2 instance IAM role
    # 2. Region from EC2 instance metadata
    # No need to pass any credentials or region explicitly
    return boto3.client('s3')


def upload_file_to_s3(
    local_file_path: str,
    s3_key: str,
    bucket_name: Optional[str] = None
) -> Optional[str]:
    """
    Upload a file to S3 bucket
    
    Args:
        local_file_path: Path to local file to upload
        s3_key: S3 object key (path in bucket)
        bucket_name: S3 bucket name (defaults to env variable)
    
    Returns:
        str: S3 URL if successful, None if failed
    """
    if bucket_name is None:
        bucket_name = os.getenv("S3_BUCKET_NAME")
    
    if not bucket_name:
        print("⚠️ S3_BUCKET_NAME not configured, skipping S3 upload")
        return None
    
    if not os.path.exists(local_file_path):
        print(f"⚠️ File not found: {local_file_path}")
        return None
    
    try:
        s3_client = get_s3_client()
        
        # Validate bucket exists and is accessible before upload
        try:
            s3_client.head_bucket(Bucket=bucket_name)
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404' or error_code == 'NoSuchBucket':
                print(f"❌ S3 bucket '{bucket_name}' does not exist")
                return None
            elif error_code == '403' or error_code == 'Forbidden':
                print(f"❌ Access denied to S3 bucket '{bucket_name}'. Check IAM permissions.")
                return None
            else:
                print(f"⚠️ Cannot access bucket '{bucket_name}': {e}")
                return None
        
        # Determine content type based on file extension
        content_type = get_content_type(local_file_path)
        
        # Upload file with metadata
        try:
            s3_client.upload_file(
                local_file_path,
                bucket_name,
                s3_key,
                ExtraArgs={'ContentType': content_type} if content_type else {}
            )
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '403' or error_code == 'AccessDenied':
                print(f"❌ Access denied: Cannot upload to s3://{bucket_name}/{s3_key}")
                print("   Check IAM role has s3:PutObject permission")
                return None
            else:
                print(f"❌ Upload failed: {e}")
                return None
        
        # Verify upload was successful by checking object existence
        try:
            s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404':
                print(f"⚠️ Upload verification failed: File not found in S3 after upload")
                return None
            else:
                print(f"⚠️ Upload verification failed: {e}")
                # Upload might have succeeded despite verification failure
                # Continue to generate URL
        
        # Generate S3 URL by automatically detecting the bucket's region
        # boto3 will retrieve the region from the bucket's location
        try:
            bucket_location = s3_client.get_bucket_location(Bucket=bucket_name)
            region = bucket_location.get('LocationConstraint')
            
            # Handle special case: eu-north-1 returns None
            if region is None:
                region = 'eu-north-1'
            
            # Generate region-specific S3 URL (permanent)
            s3_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"
        except ClientError as e:
            # Fallback to region-agnostic URL if region detection fails
            print(f"⚠️ Could not detect bucket region, using region-agnostic URL: {e}")
            s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
        
        # Generate presigned URL for temporary frontend access (30 minutes)
        try:
            presigned_url = s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': bucket_name,
                    'Key': s3_key
                },
                ExpiresIn=1800  # 30 minutes = 1800 seconds
            )
            print(f"✅ Uploaded to S3: {s3_key}")
            print(f"   Presigned URL valid for 30 minutes")
            
            # Return both permanent URL and presigned URL
            return {
                'url': s3_url,
                'presigned_url': presigned_url,
                'expires_in': 1800
            }
        except ClientError as e:
            print(f"⚠️ Could not generate presigned URL: {e}")
            # Return just the permanent URL if presigned URL generation fails
            return {
                'url': s3_url,
                'presigned_url': None,
                'expires_in': None
            }
        
    except NoCredentialsError:
        print("⚠️ AWS credentials not found, skipping S3 upload")
        print("   Ensure EC2 instance has an IAM role attached")
        return None
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        print(f"⚠️ S3 upload failed [{error_code}]: {error_message}")
        return None
    except Exception as e:
        print(f"⚠️ Unexpected error during S3 upload: {e}")
        return None


def get_content_type(file_path: str) -> Optional[str]:
    """
    Determine content type based on file extension
    
    Args:
        file_path: Path to file
    
    Returns:
        str: MIME type or None
    """
    extension = os.path.splitext(file_path)[1].lower()
    
    content_types = {
        '.pdf': 'application/pdf',
        '.json': 'application/json',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.txt': 'text/plain',
        '.log': 'text/plain',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
    }
    
    return content_types.get(extension)


def upload_pipeline_outputs(run_id: str, files_dict: dict) -> dict:
    """
    Upload multiple pipeline output files to S3 with presigned URLs
    
    Args:
        run_id: Pipeline run ID
        files_dict: Dictionary with file types as keys and local paths as values
                   Example: {'excel': '/path/to/file.xlsx', 'json': '/path/to/file.json'}
    
    Returns:
        dict: Dictionary with file types as keys and upload info (URLs, presigned URLs) as values
    """
    s3_data = {}
    
    for file_type, local_path in files_dict.items():
        if local_path and os.path.exists(local_path):
            # Create S3 key with run_id prefix
            filename = os.path.basename(local_path)
            s3_key = f"pipeline-outputs/{run_id}/{filename}"
            
            # Upload to S3
            upload_result = upload_file_to_s3(local_path, s3_key)
            
            if upload_result:
                s3_data[file_type] = upload_result
    
    return s3_data

