"""
Functions used to delete Weaviate and Qdrant vector databases.
"""

import weaviate
from weaviate.classes.init import Timeout
from qdrant_client import QdrantClient


def weaviateDeleteDB(dict_list, model):

    # Connect to your Weaviate instance
    with weaviate.connect_to_custom(
            http_host="localhost",
            http_port=5050,
            http_secure=False,
            grpc_host="localhost",
            grpc_port=50051,
            grpc_secure=False,
            additional_config=weaviate.config.AdditionalConfig(
                timeout=Timeout(init=120, query=60, insert=120)
            )
    ) as client:

        # Delete the existing collection (required to change vector type)
        if client.collections.exists("FH_PDFs"):
            client.collections.delete("FH_PDFs")


def qdrantDeleteDB(DB_name):

    client = QdrantClient(url="http://localhost:6333")

    if client.collection_exists(collection_name=DB_name):
        print(f"---{DB_name} collection exists")
        client.delete_collection(collection_name=DB_name)
        print(f"---{DB_name} collection deleted")

qdrantDeleteDB("FH_PDFs")