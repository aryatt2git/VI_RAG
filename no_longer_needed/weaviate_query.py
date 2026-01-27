import torch
import weaviate
from transformers import AutoTokenizer, AutoModel
from weaviate.classes.query import MetadataQuery

def query_RAG(RAG_query):
    client = weaviate.connect_to_custom(
        http_host="localhost",
        http_port=5050,
        http_secure=False,
        grpc_host="localhost",
        grpc_port=50051,
        grpc_secure=False,
    )

    try:
        """
        client.is_ready()
    
        # This prints out the available AIs to API to in the docker-compose.yml
        # Change the 'ENABLE_MODULES' and 'DEFAULT_VECTORIZER_MODULE' modules in docker-compose.yml as well as the API key
        # in weaviate.env.
        metainfo = client.get_meta()
        print(json.dumps(metainfo, indent=2))
        """

        # Configure collection object (The name of the vector database with the FH articles)
        chunks = client.collections.use("ArticleChunk")
        model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
        tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

        query_text = [RAG_query]
        response_text = []

        with torch.no_grad():
            # tokenize the queries
            encoded = tokenizer(
                query_text,
                truncation=True,
                padding=True,
                return_tensors='pt',
                max_length=64,
            )

            # encode the queries (use the [CLS] last hidden states as the representations)
            query_embeds = model(**encoded).last_hidden_state[:, 0, :][0].detach().cpu().numpy().tolist()

            # Perform query
            response = chunks.query.near_vector(
                near_vector=query_embeds,
                limit=5,
                return_metadata=MetadataQuery(distance=True)
            )

            # Inspect the response
            for o in response.objects:
                print(o.properties["title"])
                print(o.properties["section"])
                # print(o.properties["text"])
                print(f"Distance to query: {o.metadata.distance:.3f}\n")  # Print the distance of the object from the query

                response_text.append(o.properties["text"])

        return response_text

    finally:
        client.close()