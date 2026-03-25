from FlagEmbedding import BGEM3FlagModel
import weaviate
import weaviate.classes.config as wvc
from weaviate.classes.query import Filter, HybridFusion, MetadataQuery
from weaviate.classes.init import Timeout
from ollama import chat
import time


model = BGEM3FlagModel('BAAI/bge-m3') #, use_fp16=True

def create_context(weaviate_objects, gene, v_query):

    context = []

    for i, obj in enumerate(weaviate_objects):
        p = obj.properties
        obj_content = []

        if p["title"] != "null":
            obj_content.append(f"Title: {p['title']}\n")

        if p["authors"] != "null":
            obj_content.append(f"Authors: {p['authors']}\n")

        if p["section_header"] == "table":

            if p["subsection_header"] != "null":
                table_no = p['subsection_header']
            else:
                table_no = "number not provided."

            if p["sub_subsection_header"] != "null":
                table_section = p['sub_subsection_header']
            else:
                table_section = ""

            obj_content.append(f"Table: Table {table_no}{table_section}")

            obj_content.append(p['description'])

        elif p["section_header"] == "figure":

            if p["subsection_header"] != "null":
                fig_no = p['subsection_header']
            else:
                fig_no = "number not provided."

            if p["sub_subsection_header"] != "null":
                figure_section = p['sub_subsection_header']
            else:
                figure_section = ""

            obj_content.append(f"Figure: Figure {fig_no}{figure_section}")

            obj_content.append(p['description'])

        else:

            if p["section_header"] != "null":
                obj_content.append(f"Section: {p['section_header']}")

            if p["subsection_header"] != "null":
                obj_content.append(f"Subsection: {p['subsection_header']}")

            if p["sub_subsection_header"] != "null":
                obj_content.append(f"Paragraph: {p['sub_subsection_header']}")

            obj_content.append(f"chunk ID: {p['chunk_idx']}")
            obj_content.append(f"Content: {p['chunk']}")
            obj_content.append(f"Context: {p['description']}")



        if p.get("variants"):
            for variant in p.get("variants"):
                if gene == variant.get("gene") and v_query in [variant.get("genomic_variant"),
                                                               variant.get("transcript_variant"),
                                                               variant.get("protein_variant")
                                                               ]:
                    variant_info = []
                    for key, value in variant.items():
                        if key == "genomic_variant" and value != "null":
                            variant_info.append(f"Genomic description: {value}")
                        if key == "transcript_variant" and value != "null":
                            variant_info.append(f"Transcript description: {value}")
                        if key == "protein_variant" and value != "null":
                            variant_info.append(f"Protein description: {value}")
                        if key == "exon" and value != "null":
                            variant_info.append(f"Exon: {value}")
                        if key == "protein_domain" and value != "null":
                            variant_info.append(f"Protein domain: {value}")
                        if key == "clinical_information" and value != "null":
                            variant_info.append(f"Clinical Information from this data: {value}")
                        if key == "het_carriers" and value != "null":
                            variant_info.append(f"Number of heterozygous carriers in this paper: {value}")
                        if key == "hom_carriers" and value != "null":
                            variant_info.append(f"Number of homozygous carriers in this paper: {value}")
                        if key == "affected_carriers" and value != "null":
                            variant_info.append(f"Number of affected carriers in this paper: {value}")
                        if key == "unaffected_carriers" and value != "null":
                            variant_info.append(f"Number of unaffected carriers in this paper: {value}")
                        if key == "number_of_meioses" and value != "null":
                            variant_info.append(f"Number of meioses counted in this paper: {value}")

                    obj_content.append(f"Variant information:\n\t" + "\n\t".join(variant_info) + "\n")

        context.append("\n---------------\n\n".join(obj_content))

    return context

def hybrid_search(variant, model, gene, initial_fetch, final_fetch):

    print("---Encoding variants and gene terms")

    encoded = model.encode(variant, return_dense=True, return_sparse=True)

    variant_dense = encoded['dense_vecs'].astype("float32").tolist()

    print(f"---Dense vectors: {variant_dense[:5]}")

    variant_sparse_dict = encoded['lexical_weights']

    print(f"---Sparse vectors: {variant_sparse_dict}")
    print(f"---Connecting to weaviate database...")

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

        collection = client.collections.get("FH_PDFs")

        print(f"---Connected to weaviate database")


        total = collection.aggregate.over_all(total_count=True)
        print(f"\n---Total objects in DB: {total.total_count}")

        #filters = Filter.by_property("genes_mentioned").contains_any([gene])

        weaviate_response = collection.query.hybrid(
            query=variant,
            vector=variant_dense,
            #target_vector="dense",
            #filters=filters,
            alpha=0.5,
            limit=initial_fetch,
            fusion_type=HybridFusion.RELATIVE_SCORE,
            return_metadata=MetadataQuery(score=True),
            return_properties=[
                "title",
                "authors",
                "path",
                "section_header",
                "subsection_header",
                "sub_subsection_header",
                "chunk_idx",
                "chunk",
                "description",
                "genes_mentioned",
                "variant_count",
                #"variants",
                #"sparse_weights"
            ]
        )
        """
        uuids = []
        for obj in weaviate_response.objects:
            uuids.append(obj.uuid)

        sparse_weight_variants = {}
        for uuid in uuids:
            sparse_response = collection.query.fetch_object_by_id(
                uuid=uuid,
                return_properties= ["sparse_weights", "variants"]
            )

            if sparse_response and sparse_response.properties.get("sparse_weights"):
                sparse_weight_variants[str(uuid)] = {
                    "sparse_weights": sparse_response.properties["sparse_weights"],
                    "variants": sparse_response.properties.get("variants") or []
                }

            else:
                sparse_weight_variants[str(uuid)] = {
                    "sparse_weights": [],
                    "variants": []
                }

        scored = []
        for obj in weaviate_response.objects:

            sparse_weights = sparse_weight_variants[str(obj.uuid)]["sparse_weights"]

            obj.properties["variants"] = sparse_weight_variants[str(obj.uuid)]["variants"]

            chunk_sparse_dict = {}
            for sparse_weight in sparse_weights:
                chunk_sparse_dict[sparse_weight["token"]] = float(sparse_weight["weight"])

            score = 0.0
            for query_token, query_weight in variant_sparse_dict.items():
                query_str = str(query_token)
                if query_str in chunk_sparse_dict:
                    score += float(query_weight) * chunk_sparse_dict[query_str]

            scored.append((score, obj))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, obj in scored[:final_fetch]:
            results.append(obj)
        
        return results
        """
        results = []
        for obj in weaviate_response.objects:
            results.append(obj)

        return results

def vlm_query(variant, model, gene, initial_fetch, final_fetch):

    weaviate_response = hybrid_search(variant, model, gene, initial_fetch, final_fetch)

    if not weaviate_response:
        return {
            "answer": "Information regarding this variant could not be found.",
            "sources": [],
            "context": ""
        }

    print(f"---Found {len(weaviate_response)} chunks!")

    time_one = time.time()

    context_list = create_context(weaviate_response, gene=gene, v_query=variant)
    context = "\n\n".join(context_list)

    prompt=f"""
    Using the following source documents, please answer the following question:
    
    QUESTION: What is Familial Hypercholesterolaemia and how is it diagnosed?
    
    Sources: {context}"
    """

    messages = [
        {
            "role": "system",
            "content": "You are a genomic clinical scientist specialising in variant interpretation and classification "
                       "of genomic variants associated with Familial Hypercholesterolaemia. Use ONLY the information "
                       "provided as context to interpret the variant. Do not hallucinate. Do not speculate. If you are "
                       "unable to provide an answer, let the user know. Always provide evidence for every piece of "
                       "information you provide in your answer."

        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = chat(
        model="qwen3.5:397b-cloud",
        messages=messages,
        options={
            'temperature': 0.0,  # Lower temperature makes the output more focused and consistent
            'repeat_penalty': 1.0,  # Setting to 1.0 turns OFF the penalty that stops it from repeating itself
            'num_predict': 25000,  # Give the model plenty of room to write a long answer
            'seed': 42,  # Uncomment this if you want the EXACT same answer every single time
            'num_ctx': 32768,  # Add this to handle the image + markdown
        },
        keep_alive=0  # This forces a fresh start for the next run
    )

    print("--- VLM RESPONSE ---")

    time_two = time.time()

    print(response["message"]["content"])

    with open("ollama_query_costs.txt", 'a') as f:
        f.write(
            f"{len(weaviate_response)} chunks fed to VLM\n"
            f"vlm_query_variant: {variant}\n"
            f"Query costed: {response['prompt_eval_count']} tokens\n"
            f"Response costed: {response['eval_count']} tokens\n"
            f"Total cost: {response['prompt_eval_count'] + response['eval_count']} tokens\n"
            f"Time taken: {time_two - time_one} minutes\n\n"
        )

    print(f"Query costed: {response['prompt_eval_count']} tokens")
    print(f"Response costed: {response['eval_count']} tokens")
    print(f"Total cost: {response['prompt_eval_count'] + response['eval_count']} tokens")

    return response["message"]["content"]

answer = vlm_query(variant="diagnosis", model=model, gene="hypercholesterolaemia", initial_fetch=10, final_fetch=5)
print(answer)