# from byaldi import RAGMultiModalModel
import os
import time
import torch
from transformers import AutoProcessor, AutoModel
from pdf2image import convert_from_path, pdfinfo_from_path
from timer import timer
import weaviate
import weaviate.classes.config as wvc
from weaviate.classes.query import Filter
from weaviate.classes.init import Timeout

@timer
def upload_pdf(pdf_list):

    print('---Phase 1: Loading model---')
    start_time_1 = time.time()

    # Optionally, you can specify an `index_root`, which is where it'll save the index. It defaults to ".byaldi/".
    # RAG = RAGMultiModalModel.from_pretrained("vidore/colqwen2-v1.0")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model_id = 'nvidia/llama-nemotron-colembed-vl-3b-v2'

    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.float16
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    load_time = time.time() - start_time_1
    print(f'Model load time: {load_time:.2f} sec\n\n')

    print('---Phase 2: Prepare Weaviate Vector DB ---')

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
        # if client.collections.exists("FH_PDFs"):
        # client.collections.delete("FH_PDFs")

        # Configure the Collection for Multi-Vectors. We define it as a 'colbert' vectorizer type to handle the MaxSim logic
        if not client.collections.exists("FH_PDFs"):
            client.collections.create(
                name="FH_PDFs",
                vector_config=wvc.Configure.MultiVectors.self_provided(
                    vector_index_config=wvc.Configure.VectorIndex.hnsw(
                        distance_metric=wvc.VectorDistances.DOT  # ColBERT uses Dot Product for MaxSim
                    ),
                    # To save storage on RAM
                    encoding=wvc.Configure.VectorIndex.MultiVector.Encoding.muvera()
                ),

                # This is the key part for Late Interaction models
                properties=[
                    wvc.Property(name="title", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="page_number", data_type=wvc.DataType.INT),
                    wvc.Property(name="content", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="path", data_type=wvc.DataType.TEXT),
                ],
            )

        # Upload embeddings from your 'pdf_embeddings' variable. pdf_embeddings.shape is [num_pages, sequence_length, dim]
        collection = client.collections.get("FH_PDFs")

        print('---Phase 3: Document Encoding (Visual) and Importing---')

        upload_counter = 0

        for filepath in pdf_list:

            response = collection.query.fetch_objects(
                filters=Filter.by_property("path").equal(filepath),
                limit=1
            )

            if len(response.objects) > 0:
                print(f"Skipping {filepath} - already exists in DB.")
                upload_counter = upload_counter + 1
                continue

            print(f'Processing: {filepath}')

            pdf_path = filepath

            # Get the PDF metadata
            info = pdfinfo_from_path(pdf_path)

            print(info)

            if 'Title' not in info.keys():
                filename = os.path.splitext(os.path.basename(filepath))[0]
                info['Title'] = filename

                print(info)

            # Enter the first and last pages in the PDF.
            images = convert_from_path(pdf_path, first_page=1, last_page=info['Pages'])

            texts = [""] * len(images)

            inputs = processor(images=images, text=texts, return_tensors='pt').to(device).to(torch.float16)

            with torch.no_grad():
                pdf_embeddings = model(**inputs).last_hidden_state

            print(pdf_embeddings.shape)
            print(pdf_embeddings)

            with collection.batch.dynamic() as batch:
                for i, page_vecs in enumerate(pdf_embeddings):

                    # embedding = weaviate_embeds[i].cpu().float().tolist()  # make sure it's a numpy array
                    # Convert tensor to list for JSON serialization
                    vector_list = page_vecs.cpu().float().tolist()

                    # Create a Weaviate object
                    batch.add_object(
                        properties={
                            "title": info['Title'],
                            "page_number": i + 1,
                            "content": f"Content from page {i+1} in {info['Title']}.",
                            "path": pdf_path
                        },
                        vector={
                            "default": vector_list # Weaviate handles the multi-vector storage
                        }
                    )

            upload_counter = upload_counter + 1
            print(upload_counter)

        upload_time = time.time() - start_time_1
        print(upload_time)

pdf_list = []

References = '/Users/arjun/PycharmProjects/VI_RAG/VI_RAG/VLM_RAG/References/'

for root, dir, files in os.walk(References, topdown=False):
    for file in files:
        if file.endswith('.pdf'):
            abs_path = os.path.join(os.path.abspath(root), file)
            pdf_list.append(abs_path)

"""
for file in os.listdir(References):
    if file.endswith('.pdf'):
        abs_path = os.path.join(os.path.abspath(References), file)
        pdf_list.append(abs_path)
"""

upload_pdf(pdf_list)