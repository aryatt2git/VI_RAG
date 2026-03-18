import base64
import os
import json
from ollama import chat
from ollama import ChatResponse


def vlm_query(image_path, markdown_path):

    with open(markdown_path, "r", encoding='utf-8') as f:
        context = f.read()

    VLM_query = ('Extract genomic variants from the attached image along with the associated gene and any clinically '
                 'relevant information. Analyse the information as comprehensively as possible. Summarise the information '
                 'but do not omit any clinically relevant information. Structure your output into the following JSON format:'
                 '{"title": "", "authors": [], "type": "", "figure_table_no": "", "caption": "", "description": "", "table": "", "genes_mentioned": [], "variant_count": "", "variants": {"gene": "", "genomic_variant": "", "transcript_variant": "", "exon": "", "protein_variant": "", "protein_domain": "", "clinical_information": "", "clinical_significance": "", "evidence": "", "frequency": "", "hetero_carriers": "", "homo_carriers": "", "affected_carriers": "", "unaffected_carriers": ""}}'
                 )

    augmented_prompt = f"""
    Use the following context to answer the question. 
    ---
    CONTEXT:
    {context}
    ---
    QUESTION: {VLM_query}

    INSTRUCTIONS: 
    - Extract as much information from the attached image as possible.
    - Only use information from the attached image and the attached markdown file for context.
    - Only analyse images that are referenced inside table/figure legends/captions in the markdown file. Other images should be skipped.
    - Match the images to table/figure legends/captions in the markdown file
    - Know that the image's filename does not necessarily correspond with the image number, figure number or table number in the markdown file. Instead, compare the information from the image to the information in the figure/image/table legends/captions to find which legend/caption describes the image most accurately.
    - To string fields, return 'null' if information cannot be provided by the image/figure/table legend/caption.
    - To a list field that cannot be populated, return [].
    - Output RAW JSON only. Do not include any introductory text, markdown code blocks (```), or concluding remarks.
    - Never miss a variant.
    - Analyse the entire variant.
    - Never omit keys in the JSON. 
    - Never hallucinate.
    - Never speculate.
    - Do your best.
    - If you are unable to extract any clinically relevant information or cannot understand the information, let the user know.
    - To the 'title' key, assign the name of research paper/article.
    - To the 'authors' key, assign a list of the names of the authors that wrote the research paper/article.
    - To the 'type' key, if the image is a figure assign 'figure' but if the image is a table, assign 'table'.
    - To the 'figure_table_no' key, assign the table or figure number in the caption/legend matched with the image from the markdown file provided as context.
    - To the 'caption' key, assign the text provided in the caption/legend matched with the image from the markdown file provided as context.
    - To the 'description' key, provide and assign a comprehensive analysis and description of the clinical information and significance of the image. Use the text from the caption/legend matched with the image to support the explanation.
    - To the 'table' key, if the image is of a table, assign a textualised version of the table that can be embedded by a text embedder.
    - To the 'genes_mentioned' key, assign a list of the name of the genes mentioned in either the image or the caption/legend matched with the image from the markdown file provided as context.
    - To the 'variant_count' key, assign the number of genomic variants that appear in the image and the caption/legend matched with the image from the markdown file provided as context.
    - To the 'variants' key, assign a nested dictionary for each and every variant mentioned in either the image or the caption/legend matched with the image from the markdown file provided as context. Do not miss any variants out. 
    - To the 'gene' key, assign the gene symbol of the gene that the corresponding variant is in.
    - To the 'genomic_variant' key, assign the genomic variant as it appears in the image or the caption/legend matched with the image from the markdown file provided as context. It should start with 'g.' but might not.
    - To the 'transcript_variant' key, assign the variant described at the transcript level as it appears in the image or the caption/legend matched with the image from the markdown file provided as context. It should start with 'c.' but might not.
    - To the 'exon' key, assign the exon of the gene that the corresponding variant is in.
    - To the 'protein_variant' key, assign the variant described at the protein level as it appears in the image or the caption/legend matched with the image from the markdown file provided as context. It should start with 'p.' but might not.
    - To the 'protein_domain' key, assign the protein domain of the protein that the corresponding variant is in.
    - To the 'clinical_information' key, provide and assign a comprehensive analysis and description of the clinical information related to the corresponding variant from the image or the caption/legend matched with the image from the markdown file provided as context.
    - To the 'clinical_significance' key, provide and assign a comprehensive analysis and description of the clinical significance of the corresponding variant from the image or the caption/legend matched with the image from the markdown file provided as context.
    - To the 'evidence' key, assign the text from the image or the caption/legend matched with the image from the markdown file provided as context that was used to substantiate the clinical information and significance.
    - To the 'frequency' key, assign the frequency of the corresponding variant as described in the image or the caption/legend matched with the image from the markdown file provided as context.
    - To the 'hetero_carriers' key, assign the number of affected carriers with the corresponding variant in a heterozygous genotype counted in the image or the caption/legend matched with the image from the markdown file provided as context.
    - To the 'homo_carriers' key, assign the number of affected carriers with the corresponding variant in a homozygous genotype counted in the image or the caption/legend matched with the image from the markdown file provided as context.
    - To the 'affected_carriers' key, assign the total number of affected carriers with the corresponding variant counted in the image or the caption/legend matched with the image from the markdown file provided as context.
    - To the 'unaffected_carriers' key, assign the total number of unaffected carriers with the corresponding variant counted in the image or the caption/legend matched with the image from the markdown file provided as context.
    """

    response: ChatResponse = chat(
        model="qwen3.5:397b-cloud",
        messages=[
            {
                'role': 'system',
                'content': 'You are the best genomic clinical scientist ever because of your ability to understand '
                           'information. Use only provided evidence for clinical assertions. Do not hallucinate. '
                           'Do not speculate.'
            },
            {
                'role': 'user',
                'content': augmented_prompt,
                'images': [image_path]
            }
        ],
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
    # print(response['message']['content'])
    # or access fields directly from the response object
    print(response.message.content)
    print(f"Query costed: {response.prompt_eval_count} tokens")
    print(f"Response costed: {response.eval_count} tokens")
    print(f"Total cost: {response.prompt_eval_count + response.eval_count} tokens")

    return response.message.content



images = '/Users/arjun/PycharmProjects/VI_RAG/VI_RAG/tools/images/Hori_et_al_2019_PMID_31491741/'

for file in os.listdir(images):
    print(file)
    if file.endswith('4.png'):
        filename = file.split('-')[0]
        markdown_path = os.path.join(images, '..', '..', f'{filename}.md')
        print(markdown_path)
        image_path = os.path.join(images, file)
        print(f"Processing: {image_path}")

        variant_count = 0
        image_dict = None
        for attempt in range(5):
            response = vlm_query(image_path=image_path, markdown_path=markdown_path)

            try:
                resp_dict = json.loads(response)
            except:
                continue

            count = int(resp_dict["variant_count"])
            print(count)

            if count > variant_count:
                image_dict = resp_dict
            else:
                continue

        print(image_dict)