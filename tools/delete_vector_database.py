import weaviate
import weaviate.classes.config as wvc
from weaviate.classes.query import Filter
from weaviate.classes.init import Timeout


def weaviateImportText(dict_list, model):

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