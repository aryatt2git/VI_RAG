import torch
import json
import numpy as np
from FlagEmbedding import BGEM3FlagModel
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
        # if client.collections.exists("FH_PDFs"):
        # client.collections.delete("FH_PDFs")

        # Configure the Collection for Multi-Vectors. We define it as a 'colbert' vectorizer type to handle the MaxSim logic
        if not client.collections.exists("FH_PDFs"):
            client.collections.create(
                name="FH_PDFs",
                vector_config=[
                    wvc.Configure.Vectors.self_provided(
                        name="dense",
                        vector_index_config=wvc.Configure.VectorIndex.hnsw(
                            distance_metric=wvc.VectorDistances.COSINE  # ColBERT uses Dot Product for MaxSim
                        ),
                    ),
                    wvc.Configure.Vectors.none(
                        name="sparse",
                    )
                ],
                inverted_index_config=wvc.Configure.inverted_index(
                    bm25_b=0.7,
                    bm25_k1=1.25,
                    indexTimestamps=True,
                    indexNullState=True,
                    invertedIndexConfig=True
                ),
                # This is the key part for Late Interaction models
                properties=[
                    wvc.Property(name="title", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="authors", data_type=wvc.DataType.TEXT_ARRAY),
                    wvc.Property(
                        name="path",
                        data_type=wvc.DataType.TEXT,
                        tokenization=wvc.config.Tokenization.FIELD
                    ),

                    wvc.Property(
                        name="section_header",
                        data_type=wvc.DataType.TEXT,
                        tokenization=wvc.config.Tokenization.FIELD
                    ),
                    wvc.Property(
                        name="subsection_header",
                        data_type=wvc.DataType.TEXT,
                        tokenization=wvc.config.Tokenization.FIELD
                    ),
                    wvc.Property(
                        name="sub_subsection_header",
                        data_type=wvc.DataType.TEXT,
                        tokenization=wvc.config.Tokenization.FIELD
                    ),

                    wvc.Property(name="chunk_idx", data_type=wvc.DataType.INT),

                    wvc.Property(name="chunk", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="description", data_type=wvc.DataType.TEXT),

                    wvc.Property(name="genes_mentioned", data_type=wvc.DataType.TEXT_ARRAY),
                    wvc.Property(name="variant_count", data_type=wvc.DataType.INT),

                    #wvc.Property(name="sparse_weights", data_type=wvc.DataType.TEXT),

                    wvc.Property(name="variants", data_type=wvc.DataType.TEXT_ARRAY)
                ]
            )

        # Upload embeddings from your 'pdf_embeddings' variable. pdf_embeddings.shape is [num_pages, sequence_length, dim]
        collection = client.collections.get("FH_PDFs")

        print(f'---Encoding and Importing chunks---')

        upload_counter = 0

        with collection.batch.dynamic() as batch:

            for var_dict in dict_list:

                response = collection.query.fetch_objects(
                    filters=Filter.by_property("path").equal(var_dict["path"]) &
                            Filter.by_property("section_header").equal(var_dict["section_header"]) &
                            Filter.by_property("subsection_header").equal(var_dict["subsection_header"]) &
                            Filter.by_property("sub_subsection_header").equal(var_dict["sub_subsection_header"]) &
                            Filter.by_property("chunk_idx").equal(var_dict["chunk_idx"]),
                    limit=1
                )

                if len(response.objects) > 0:
                    print(f"Skipping {var_dict['path']} - already exists in DB.")
                    continue

                encoded = model.encode(var_dict["chunk"], return_dense=True, return_sparse=True)

                dense = encoded['dense_vecs'].astype("float32").tolist()

                indices = []
                values = []
                sparse_dict = encoded['lexical_weights']
                for key, value in sparse_dict.items():
                    indices.append(int(key))
                    values.append(float(value))

                print(indices)
                print(values)

                with open("chunk_sizes.txt", "a") as f:
                    f.write(
                        f"Path: {var_dict['path']}\n"
                        f"Paper: {var_dict['title']}\n"
                        f"Section: {var_dict['section_header']}\n"
                        f"Subsection: {var_dict['subsection_header']}\n"
                        f"Chunk No.: {var_dict['chunk_idx']}\n"
                        f"Dense vector length: {len(dense)}\n"
                        f"Sparse indices length: {len(indices)}\n"
                        f"Sparse values length: {len(values)}\n\n"
                    )

                properties = {
                    "title": var_dict["title"],
                    "authors": var_dict["authors"],
                    "path": var_dict["path"],
                    "section_header": var_dict["section_header"],
                    "subsection_header": var_dict["subsection_header"],
                    "sub_subsection_header": var_dict["sub_subsection_header"],
                    "chunk_idx": var_dict["chunk_idx"],
                    "chunk": var_dict["chunk"],
                    "description": var_dict["description"],
                    "genes_mentioned": var_dict["genes_mentioned"],
                    "variant_count": int(var_dict["variant_count"]),
                    #"sparse_weights": sparse,
                    "variants": [json.dumps(variant) for variant in var_dict.get("variants", [])]
                }

                batch.add_object(
                    properties=properties,
                    vector={
                        "dense": dense,
                        "sparse": {
                            "indices": indices,
                            "values": values
                        }
                    }
                )

                upload_counter = upload_counter + 1

            print(f"---{upload_counter} chunks uploaded.")

            if collection.batch.failed_objects:
                print(f"---Failed to import {len(collection.batch.failed_objects)} objects.")
                for obj in collection.batch.failed_objects:
                    print(f"---Error: {obj.message}")


def weaviateImportImage(dict_list, model):

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
                vector_config=[
                    wvc.Configure.Vectors.self_provided(
                        name="dense",
                        vector_index_config=wvc.Configure.VectorIndex.hnsw(
                            distance_metric=wvc.VectorDistances.COSINE  # ColBERT uses Dot Product for MaxSim
                        ),
                    )
                ],
                # This is the key part for Late Interaction models
                properties=[
                    wvc.Property(name="title", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="authors", data_type=wvc.DataType.TEXT_ARRAY),
                    wvc.Property(
                        name="path",
                        data_type=wvc.DataType.TEXT
                    ),

                    wvc.Property(
                        name="section_header",
                        data_type=wvc.DataType.TEXT,
                        tokenization=wvc.config.Tokenization.FIELD
                    ),
                    wvc.Property(
                        name="subsection_header",
                        data_type=wvc.DataType.TEXT,
                        tokenization=wvc.config.Tokenization.FIELD
                    ),
                    wvc.Property(
                        name="sub_subsection_header",
                        data_type=wvc.DataType.TEXT,
                        tokenization=wvc.config.Tokenization.FIELD
                    ),

                    wvc.Property(name="chunk_idx", data_type=wvc.DataType.INT),

                    wvc.Property(name="chunk", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="description", data_type=wvc.DataType.TEXT),

                    wvc.Property(name="genes_mentioned", data_type=wvc.DataType.TEXT_ARRAY),
                    wvc.Property(name="variant_count", data_type=wvc.DataType.INT),

                    wvc.Property(
                        name="sparse_weights",
                        data_type=wvc.DataType.OBJECT_ARRAY,
                        nested_properties=[
                            wvc.Property(name="token", data_type=wvc.DataType.TEXT),
                            wvc.Property(name="weight", data_type=wvc.DataType.NUMBER),
                        ]
                    ),

                    wvc.Property(
                        name="variants",
                        data_type=wvc.DataType.OBJECT_ARRAY,
                        nested_properties = [
                            wvc.Property(name="gene", data_type=wvc.DataType.TEXT),
                            wvc.Property(name="genomic_variant", data_type=wvc.DataType.TEXT),
                            wvc.Property(name="transcript_variant", data_type=wvc.DataType.TEXT),
                            wvc.Property(name="exon", data_type=wvc.DataType.TEXT),
                            wvc.Property(name="protein_variant", data_type=wvc.DataType.TEXT),
                            wvc.Property(name="protein_domain", data_type=wvc.DataType.TEXT),
                            wvc.Property(name="clinical_information", data_type=wvc.DataType.TEXT),
                            wvc.Property(name="clinical_significance", data_type=wvc.DataType.TEXT),
                            wvc.Property(name="frequency", data_type=wvc.DataType.TEXT),
                            wvc.Property(name="het_carriers", data_type=wvc.DataType.TEXT),
                            wvc.Property(name="hom_carriers", data_type=wvc.DataType.TEXT),
                            wvc.Property(name="affected_carriers", data_type=wvc.DataType.TEXT),
                            wvc.Property(name="unaffected_carriers", data_type=wvc.DataType.TEXT),
                            wvc.Property(name="number_of_meioses", data_type=wvc.DataType.TEXT)
                        ]
                    )
                ]
            )

        # Upload embeddings from your 'pdf_embeddings' variable. pdf_embeddings.shape is [num_pages, sequence_length, dim]
        collection = client.collections.get("FH_PDFs")

        print(f'---Encoding and Importing chunks---')

        upload_counter = 0

        with collection.batch.dynamic() as batch:

            for var_dict in dict_list:

                response = collection.query.fetch_objects(
                    filters=Filter.by_property("path").equal(var_dict["path"]) &
                            Filter.by_property("section_header").equal(var_dict["type"]) &
                            Filter.by_property("subsection_header").equal(var_dict["figure_table_no"]) &
                            Filter.by_property("sub_subsection_header").equal(var_dict["sub_figure_table_no"]) &
                            Filter.by_property("chunk_idx").equal(var_dict["chunk_idx"]),
                    limit=1
                )

                if len(response.objects) > 0:
                    print(f"Skipping {var_dict['path']} - already exists in DB.")
                    continue

                encoded = model.encode(var_dict["chunk"], return_dense=True, return_sparse=True)

                dense = encoded['dense_vecs'].astype("float32").tolist()

                sparse_dict = encoded['lexical_weights']
                sparse = []
                for key, value in sparse_dict.items():
                    sparse_weight = {}
                    sparse_weight["token"] = key
                    sparse_weight["weight"] = float(value)
                    sparse.append(sparse_weight)

                with open("chunk_sizes.txt", "a") as f:
                    f.write(f"Chunk No.: {var_dict['chunk_idx']}\n"
                            f"Dense vector length: {len(dense)}\n"
                            f"Sparse vector length: {len(sparse)}\n")

                properties = {
                    "title": var_dict["title"],
                    "authors": var_dict["authors"],
                    "path": var_dict["path"],
                    "section_header": var_dict["type"],
                    "subsection_header": var_dict["figure_table_no"],
                    "sub_subsection_header": var_dict["sub_figure_table_no"],
                    "chunk_idx": var_dict["chunk_idx"],
                    "chunk": var_dict["chunk"],
                    "description": f"Caption: {var_dict['caption']} \nDescription: {var_dict['description']} \nTable: {var_dict['table']}",
                    "genes_mentioned": var_dict["genes_mentioned"],
                    "variant_count": int(var_dict["variant_count"]),
                    "sparse_weights": sparse,
                    "variants": [
                        {
                            "gene": variant["gene"],
                            "genomic_variant": variant["genomic_variant"],
                            "transcript_variant": variant["transcript_variant"],
                            "exon": variant["exon"],
                            "protein_variant": variant["protein_variant"],
                            "protein_domain": variant["protein_domain"],
                            "clinical_information": variant["clinical_information"],
                            "clinical_significance": variant["clinical_significance"],
                            "frequency": variant["frequency"],
                            "het_carriers": variant["het_carriers"],
                            "hom_carriers": variant["hom_carriers"],
                            "affected_carriers": variant["affected_carriers"],
                            "unaffected_carriers": variant["unaffected_carriers"],
                            "number_of_meioses": variant["number_of_meioses"]
                        } for variant in var_dict.get("variants", [])
                    ]
                }

                batch.add_object(
                    properties=properties,
                    vector=dense
                )

                upload_counter = upload_counter + 1

            print(f"---{upload_counter} chunks uploaded.")

            if collection.batch.failed_objects:
                print(f"---Failed to import {len(collection.batch.failed_objects)} objects.")
                for obj in collection.batch.failed_objects:
                    print(f"---Error: {obj.message}")