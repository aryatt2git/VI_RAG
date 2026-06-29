from FlagEmbedding import BGEM3FlagModel
from qdrant_functions import qdrantHybridSearch
from ollama import chat
import time


model = BGEM3FlagModel('BAAI/bge-m3') #, use_fp16=True

def create_context(qdrant_response, gene, v_query):

    context = []

    for i, point in enumerate(qdrant_response.points):

        payload = point.payload

        obj_content = []

        if payload["title"] != "null":
            obj_content.append(f"Title: {payload['title']}\n")

        if payload["authors"] != "null":
            obj_content.append(f"Authors: {payload['authors']}\n")

        if "type" in payload:

            if payload["type"] == "table":

                if payload["figure_table_no"] != "null":
                    table_no = payload["figure_table_no"]
                else:
                    table_no = "number not provided."

                if payload["sub_figure_table_no"] != "null":
                    table_section = payload['sub_figure_table_no']
                else:
                    table_section = ""

                obj_content.append(f"Table: Table {table_no}{table_section}\n")

                description = f"Caption: {payload['caption']}\n\nDescription: {payload['description']}\n\nTable: {payload['table']}\n"

                obj_content.append(description)

            elif payload["type"] == "figure":

                if payload["figure_table_no"] != "null":
                    fig_no = payload['figure_table_no']
                else:
                    fig_no = "number not provided."

                if payload["sub_figure_table_no"] != "null":
                    figure_section = payload['sub_figure_table_no']
                else:
                    figure_section = ""

                obj_content.append(f"Figure: Figure {fig_no}{figure_section}\n")

                description = f"Caption: {payload['caption']}\n\nDescription: {payload['description']}\n"

                obj_content.append(description)

        else:

            if payload["section_header"] != "null":
                obj_content.append(f"Section: {payload['section_header']}\n")

            if payload["subsection_header"] != "null":
                obj_content.append(f"Subsection: {payload['subsection_header']}\n")

            if payload["sub_subsection_header"] != "null":
                obj_content.append(f"Paragraph: {payload['sub_subsection_header']}\n")

            obj_content.append(f"chunk ID: {payload['chunk_idx']}\n")
            obj_content.append(f"Content: {payload['chunk']}\n")
            obj_content.append(f"Description: {payload['description']}\n")

        """
        if payload.get("variants"):
            for variant in payload.get("variants"):
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
        """
        context.append("\n".join(obj_content))

    response = "\n\n---------------------------------------\n\n".join(context)

    return response

def vlm_query(variant, model, gene):

    time_one = time.time()

    qdrant_response = qdrantHybridSearch(query=variant, model=model)

    if not qdrant_response:
        return {
            "answer": "Information regarding this variant could not be found.",
            "sources": [],
            "context": ""
        }

    print(f"---Found {len(qdrant_response.points)} chunks!")

    context = create_context(qdrant_response, gene=gene, v_query=variant)
    #context = "\n\n".join(context_list)

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
            f"{len(qdrant_response.points)} chunks fed to VLM\n"
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

vlm_query(variant="diagnosis", model=model, gene="hypercholesterolaemia")
