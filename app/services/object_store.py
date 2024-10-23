import boto3
from app.core.config import get_settings
from langchain_community.document_loaders import S3FileLoader

settings = get_settings()


class ObjectStore:
    def __init__(self):
        self.conn = self.get_obj_store_connection()

    def get_obj_store_connection(self):
        session = boto3.session.Session()
        return session.client(
            service_name="s3",
            endpoint_url=settings.SCW_BUCKET_ENDPOINT,
            aws_access_key_id=settings.SCW_ACCESS_KEY,
            aws_secret_access_key=settings.SCW_SECRET_KEY,
        )

    def get_page_iterator(self): 

        paginator = self.conn.get_paginator("list_objects_v2")
        return paginator.paginate(Bucket=settings.SCW_BUCKET_NAME)
    

    def get_document (self, obj): 
        file_loader = S3FileLoader(
            bucket=settings.SCW_BUCKET_NAME,
            key=obj["Key"],
            endpoint_url=settings.SCW_BUCKET_ENDPOINT,
            aws_access_key_id=settings.SCW_ACCESS_KEY,
            aws_secret_access_key=settings.SCW_SECRET_KEY,
        )
        file_to_load = file_loader.load()
        return file_to_load[0].page_content