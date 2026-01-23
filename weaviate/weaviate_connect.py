import json
import weaviate

client = weaviate.connect_to_custom(
    http_host="localhost",
    http_port=5000,
    http_secure=False,
    grpc_host="localhost",
    grpc_port=50051,
    grpc_secure=False,
)

try:
    client.is_ready()

    # This prints out the available AIs to API to in the docker-compose.yml
    # Change the 'ENABLE_MODULES' and 'DEFAULT_VECTORIZER_MODULE' modules in docker-compose.yml as well as the API key
    # in weaviate.env.
    metainfo = client.get_meta()
    print(json.dumps(metainfo, indent=2))

finally:
    client.close()