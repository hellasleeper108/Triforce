from minio import Minio
from minio.error import S3Error
import os
import io

class StorageClient:
    def __init__(self, endpoint=None, access_key=None, secret_key=None, bucket_name="triforce-jobs"):
        self.endpoint = endpoint or os.getenv("MINIO_ENDPOINT", "minio:9000")
        self.access_key = access_key or os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = secret_key or os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.bucket_name = bucket_name
        
        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=False
        )
        self._ensure_bucket()

    def _ensure_bucket(self):
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
        except Exception as e:
            print(f"Error checking bucket: {e}")

    def upload_bytes(self, data: bytes, object_name: str) -> str:
        """Uploads bytes and returns the object name/path"""
        try:
            self.client.put_object(
                self.bucket_name,
                object_name,
                io.BytesIO(data),
                len(data)
            )
            return object_name
        except S3Error as e:
            raise Exception(f"Failed to upload {object_name}: {e}")

    def download_bytes(self, object_name: str) -> bytes:
        """Downloads object as bytes"""
        try:
            response = self.client.get_object(self.bucket_name, object_name)
            return response.read()
        except S3Error as e:
             raise Exception(f"Failed to download {object_name}: {e}")
        finally:
            if 'response' in locals():
                response.close()
                
    def get_presigned_url(self, object_name: str, method="GET") -> str:
        # Note: Internal endpoint might not be reachable by client if separate network
        # For internal cluster usage, direct download is fine.
        return self.client.get_presigned_url(method, self.bucket_name, object_name)
