import torch
import weaviate
from transformers import AutoProcessor, AutoModel
from weaviate.classes.query import MetadataQuery

import os
from pdf2image import convert_from_path

def display_pdf_page(pdf_path, page_num):
    """
    Renders a single page of a PDF as an image.
    Note: pdf2image uses 1-based indexing for pages.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: Could not find file at {pdf_path}")
        return

    print(f"--- Rendering Page {page_num} of {os.path.basename(pdf_path)} ---")

    # Convert only the specific page
    images = convert_from_path(
        pdf_path,
        first_page=page_num,
        last_page=page_num,
        dpi=200  # Higher DPI = clearer text but slower rendering
    )

    if images:

        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

        img = images[0]
        # If in a Jupyter Notebook/VS Code Notebook:
        # display(img)

        # Save for later
        dir = os.path.dirname(os.path.abspath(__file__))
        save_pdf_as = f"{pdf_name}_page_{page_num}.png"
        save_path = os.path.join(dir, "context", save_pdf_as)
        img.save(save_path)

        return {"pdfFile": pdf_name, "pdfPath": save_path}

    else:
        print("Failed to render page.")
        return None

def query_PDF_RAG(RAG_query):
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
        collection = client.collections.use("FH_PDFs")

        model_id = 'nvidia/llama-nemotron-colembed-vl-3b-v2'

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=torch.float16
        ).to(device)

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        query_text = [RAG_query]
        evidence = []

        with torch.no_grad():
            # tokenize the queries
            input = processor(
                text=query_text,
                return_tensors='pt'
            ).to(device).to(torch.float16)

            # encode the queries (use the [CLS] last hidden states as the representations)
            query_embeds = model(**input).last_hidden_state

            query_vector = query_embeds[0].cpu().float().tolist()

            # Perform query
            response = collection.query.hybrid(
                query=RAG_query,
                vector=query_vector,
                alpha=0.5,
                limit=3,
                # Use distance instead of score when looking for semantic distance. Score is used for hybrid searches.
                return_metadata=MetadataQuery(score=True)
            )

            if not response.objects:
                print("No relevant documents found.")
                return None

            # Inspect the response
            for o in response.objects:
                pdf = display_pdf_page(o.properties['path'], o.properties['page_number'])

                file_name = pdf['pdfFile']
                image = pdf['pdfPath']

                print(f"Paper: {file_name}")
                print(f"Paper: {image}")
                print(f"Score: {o.metadata.score:.4f}\n")  # Print the distance of the object from the query

                evidence.append(image)

        return evidence

    finally:
        client.close()