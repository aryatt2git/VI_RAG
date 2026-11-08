from ollama import chat
import os
from logger import setup_logging
import logging
import datetime

# Setup logging.
setup_logging()
logger = logging.getLogger(__name__)

#model = "gemma4:31b-cloud" # Context window not large enough
#model="beam:cloud" # Does not accept images
#model="deepseek-v4.1-flash:1017-cloud"
#model="glm-5.3-flash:cloud"
#model="kimi-k3:cloud"



def variant_text_extraction(markdown):

    context = ""
    with open(markdown, "r") as f:
        context = f.read()

    LLM_query = ('Return a JSON object containing the title and authors of the attached markdown file. The JSON should '
                 'also contain a list of every variant associated with Familial Hypercholesterolaemia that appears or '
                 'is mentioned in the markdown file. Only use the attached markdown file for context. Each variant '
                 'should appear in the list only once. Different descriptions of the same variant should be included '
                 'in the same entry in the list. Do not miss any variants out. Do not hallucinate. Do not speculate. '
                 'Only return a list; do not include any additional comments. The list should be contained in square '
                 'brackets so that it can be processed in Python. The JSON should have the following format:'
                 '{'
                 '  "title": ""'
                 '  "authors": []'
                 '  "variants": []'
                 '}'
                 'Never omit keys in the JSON. To string fields, enter the text "null" in the string field if '
                 'information cannot be provided. To a list field that cannot be populated, return [].')

    prompt = f"""
    ---------------------
    CONTEXT: {context}
    ---------------------
    QUESTION: {LLM_query}
    ---------------------
    """

    messages = [
        {
            "role": "system",
            "content": "You are a genomic clinical scientist with the ability to understand, analyse and interpret the "
                       "content of research papers, articles, and literature stored in markdown format. Use only "
                       "content from the markdown files for context. Do not hallucinate. Do not speculate."
        },
        {
            "role": "user",
            "content": f"{prompt}"
        }
    ]

    # The chat() function sends the above message to a specified model with additional parameters. The response is
    # returned in JSON.
    response = chat(
        model="gemma4:31b-cloud",
        messages=messages,
        format="json",
        options={
            'temperature': 0.0,  # 0-1. Lower temperature makes the output more focused and consistent
            'repeat_penalty': 1.0,  # Setting to 1.0 turns OFF the penalty that stops it from repeating itself.
            # >1.0 increases the penalty for repetition, making repetition less likely.
            'num_predict': 25000,  # Give the model plenty of room to write a long answer
            'seed': 42,  # This combines with temperature to produce the EXACT same answer every single time.
            'num_ctx': 32768,  # Add this to handle the image + markdown
        }
    )

    # Log the VLM response to keep a record of what was returned by the variant_text_extraction function.
    logger.info(
        f"variant_text_extraction() response for {os.path.basename(markdown)}:\n{response['message']['content']}")

    # The path to the markdown file, the file size of the markdown file, the number of tokens it cost to query and
    # receive a response from the VLM are all recorded in the 'ollama_token_costs.txt' file, in case the token cost
    # needs to be evaluated.
    with open("ollama_token_costs.txt", 'a') as f:
        f.write(f"{str(datetime.datetime.now())}\n"
                f"Model: gemma4:31b-cloud\n"
                f"variant_text_extraction()\n"
                f"Path to markdown file processed by VLM: {markdown}\n"
                f"Size of markdown file: {os.path.getsize(markdown)} bytes\n"
                f"Query costed: {response['prompt_eval_count']} tokens\n"
                f"Response costed: {response['eval_count']} tokens\n"
                f"Total cost: {response['prompt_eval_count'] + response['eval_count']} tokens\n\n")

    # Log the number of tokens it cost to generate the response.
    logger.info(f"Query costed: {response['prompt_eval_count']} tokens")
    logger.info(f"Response costed: {response['eval_count']} tokens")
    logger.info(f"Total cost: {response['prompt_eval_count'] + response['eval_count']} tokens")

    # The JSON response is returned.
    return response["message"]["content"]

def variant_text_description(variant, markdown):

    context = ""
    with open(markdown, "r") as f:
        context = f.read()

    LLM_query = (f'Analyse {variant} using only the attached markdown file for context. Return the outcome from the '
                 f'analysis as a JSON object in the following format:'
                 '{'
                 '  "gene": "",'
                 '  "exon": "",'
                 '  "genomic_variant": "",'
                 '  "transcript_variant": "",'
                 '  "protein_variant": "",'
                 '  "protein_domain": "",'
                 '  "inclusion": "",'
                 '  "laboratory": "",'
                 '  "patient_symptoms": "",'
                 '  "clinical_information": "",'
                 '  "clinical_significance": "",'
                 '  "evidence": "",'
                 '  "frequency": "",'
                 '  "het_carriers": "",'
                 '  "hom_carriers": "",'
                 '  "affected_carriers": "",'
                 '  "unaffected_carriers": "",'
                 '  "number_of_meioses": "",'
                 '  "n_mentions": "",'
                 '} '
                 'Populate the JSON using the following instructions:')

    prompt = f"""
    ---------------------
    CONTEXT: {context}
    ---------------------
    QUESTION: {LLM_query}
    ---------------------
    INSTRUCTIONS:
    - Only analyse {variant}
    - Extract as much information from the markdown file as possible.
    - Use all of the content from the markdown file for context.
    - Never hallucinate.
    - Never speculate.
    - Only use information from the markdown file for context.
    - Do not include any or content from image legends or image captions or image annotations or table legends or table captions or table annotations or figure legends or figure captions or figure annotations from the markdown file content in the output.
    - Do not include any or content from images or figures or tables from the markdown file content in the output.
    - Match lines that end abruptly to other lines in the same section that start abruptly and check that they make sense.
    - To string fields, enter the text "null" in the string field if information cannot be provided by the image legend or image caption or table legend or table caption or figure legend or figure caption or markdown file.
    - To a list field that cannot be populated, return [].
    - Output RAW JSON only. Do not include any introductory text, markdown code blocks (```), or concluding remarks.
    - If you see ' &gt; ' in the text, replace it with '>'. Make sure the whitespaces either side of &gt; are removed.
    - Convert special markdown syntax into plain text characters.
    - Never miss a variant.
    - Analyse the entire variant.
    - Never omit keys in the JSON. 
    - Never hallucinate.
    - Never speculate.
    - If you are unable to extract any clinically relevant information or cannot understand the information, let the user know.  
    - To the 'gene' key, assign the gene symbol of the gene that the corresponding variant is in. Use the entire markdown file for context.
    - To the 'genomic_variant' key, assign the genomic variant as it appears in the markdown file. It should start with 'g.' but might not. Remove any whitespace in the variant nomenclature.
    - To the 'transcript_variant' key, assign the variant described at the transcript level as it appears in the markdown file. It should start with 'c.' but might not. Remove any whitespace in the variant nomenclature.
    - To the 'exon' key, assign the exon of the gene that the corresponding variant is in. Use the entire markdown file for context.
    - To the 'protein_variant' key, assign the variant described at the protein level as it appears in the markdown file. It should start with 'p.' but might not. Remove any whitespace in the variant nomenclature.
    - To the 'protein_domain' key, assign the protein domain of the protein that the corresponding variant is in. Use the entire markdown file for context.
    - To the 'inclusion' key, assign a detailed description of how the variant was included in the markdown file, including how patients were recruited into the study and the method by which the variant was identified, and any other information that is relevant. Include a description of the precise characteristics that were used to define clinical FH in patients.
    - To the 'laboratory' key, assign the laboratory where the variant was identified, where the patient was tested and the country that the laboratories are in.
    - To the 'patient_information' key, assign the clinical information, symptoms and country of origin of each patient carrying the variant mentioned in the markdown file.
    - To the 'clinical_information' key, provide and assign a comprehensive and profound analysis and description of all of the the clinical information related to the corresponding variant from the markdown file provided as context.
    - To the 'clinical_significance' key, provide and assign a comprehensive analysis and description of the clinical significance of the corresponding variant with in the wider context provided by the markdown file.
    - To the 'evidence' key, assign all of the the text from the markdown file that was used to substantiate the clinical information and significance. Please quote the text in the markdown file, word for word, and convert any markdown syntax in the text into plain text characters.
    - To the 'frequency' key, assign the frequency of the corresponding variant as described in the markdown file provided as context.
    - To the 'het_carriers' key, assign the number of affected carriers with the corresponding variant in a heterozygous genotype counted in the markdown file provided as context.
    - To the 'hom_carriers' key, assign the number of affected carriers with the corresponding variant in a homozygous genotype counted in the markdown file provided as context.
    - To the 'affected_carriers' key, assign the total number of affected carriers with the corresponding variant counted in the markdown file provided as context.
    - To the 'unaffected_carriers' key, assign the total number of unaffected carriers with the corresponding variant counted in the markdown file provided as context.
    - One meiosis is the inheritance of a variant from one affected individual to their affected child.
    - To the 'number_of_meioses' key, assign the total number of meioses of the corresponding variant counted in the markdown file provided as context.
    - To the 'n_mentions' key, assign the total number of times the variant was mentioned in the markdown file, in any way, whether it be in its genomic, transcript or protein description or in an abbreviated format.
    """

    messages = [
        {
            "role": "system",
            "content": "You are a genomic clinical scientist with the ability to understand, analyse and interpret the "
                       "content of research papers, articles, and literature stored in markdown format. Use only "
                       "content from the markdown files for context. Do not hallucinate. Do not speculate."
        },
        {
            "role": "user",
            "content": f"{prompt}"
        }
    ]

    # The chat() function sends the above message to a specified model with additional parameters. The response is
    # returned in JSON.
    response = chat(
        model = "deepseek-v4.1-flash:1017-cloud",
        messages = messages,
        format="json",
        options={
            'temperature': 0.0,     # 0-1. Lower temperature makes the output more focused and consistent
            'repeat_penalty': 1.0,  # Setting to 1.0 turns OFF the penalty that stops it from repeating itself.
                                    # >1.0 increases the penalty for repetition, making repetition less likely.
            'num_predict': 25000,   # Give the model plenty of room to write a long answer
            'seed': 42,             # This combines with temperature to produce the EXACT same answer every single time.
            'num_ctx': 1000000,       # Add this to handle the image + markdown
        }
    )

    # Log the VLM response to keep a record of what was returned by the variant_text_description function.
    logger.info(f"variant_text_description() response for {os.path.basename(markdown)}:\n{response['message']['content']}")

    # The path to the markdown file, the file size of the markdown file, the number of tokens it cost to query and
    # receive a response from the VLM are all recorded in the 'ollama_token_costs.txt' file, in case the token cost
    # needs to be evaluated.
    with open("ollama_token_costs.txt", 'a') as f:
        f.write(f"{str(datetime.datetime.now())}\n"
                f"Model: deepseek-v4.1-flash:1017-cloud\n"
                f"variant_text_description()\n"
                f"Path to markdown file processed by VLM: {markdown}\n"
                f"Size of markdown file: {os.path.getsize(markdown)} bytes\n"
                f"Query costed: {response['prompt_eval_count']} tokens\n"
                f"Response costed: {response['eval_count']} tokens\n"
                f"Total cost: {response['prompt_eval_count'] + response['eval_count']} tokens\n\n")

    # Log the number of tokens it cost to generate the response.
    logger.info(f"Query costed: {response['prompt_eval_count']} tokens")
    logger.info(f"Response costed: {response['eval_count']} tokens")
    logger.info(f"Total cost: {response['prompt_eval_count'] + response['eval_count']} tokens")

    # The JSON response is returned.
    return response["message"]["content"]


def variant_image_extraction(image, markdown):

    context = ""
    with open(markdown, "r") as f:
        context = f.read()

    LLM_query = ('Return a JSON object that contains two keys: "caption" and "variants". The JSON should have the '
                 'following format:'
                 '{'
                 '  "caption": "",'
                 '  "variants": [],'
                 '}'
                 'Populate the JSON using the following instructions.')

    prompt = f"""
    ---------------------
    CONTEXT: {context}
    ---------------------
    QUESTION: {LLM_query}
    ---------------------
    INSTRUCTIONS:
    - Do not hallucinate.
    - Do not speculate. 
    - Match the attached image to an image legend or image caption or table legend or table caption or figure legend or figure caption in the markdown file, if a caption or legend was provided as context. 
    - Compare the information from the image to the information in the image legend or image caption or table legend or table caption or figure legend or figure caption to find which legend or caption describes the image most accurately.
    - To the "caption" key, assign the caption or legend that most likely describes the image.
    - If a caption or legend is unavailable or if the markdown file is empty, analyse the image in isolation and assign "null" to the "caption" key. 
    - Analyse the entire image. 
    - To the "variants" key, assign a list of every variant associated with Familial Hypercholesterolaemia that appears or is mentioned in the image or in the legend or caption related to the image from the markdown file.
    - Only extract variants from the image and associated caption or associated legend from the markdown file. Do not extract variants from the remainder of the markdown file.
    - Only use the attached markdown file for context.
    - Report every variant.
    - Never miss a variant.
    - Each variant should appear in the list only once.
    - Different descriptions of the same variant should be included in the same entry in the list.
    - Do not omit any variants.
    - Only include variants in the list; do not include any additional comments.
    - Only return a JSON object; do not include any additional comments.
    - Each entry in the list should be a string. The list should be contained in square brackets so that it can be processed in Python.
    - If there are non variants in the image or associated caption or associated legend, assign an empty list to the "variants" key.
    """

    messages = [
        {
            "role": "system",
            "content": "You are a genomic clinical scientist with the ability to understand, analyse and interpret the "
                       "content of research papers, articles, and literature stored in markdown format. Use only "
                       "content from the markdown files for context. Do not hallucinate. Do not speculate."
        },
        {
            "role": "user",
            "content": f"{prompt}",
            "images": [image]
        }
    ]

    # The chat() function sends the above message to a specified model with additional parameters. The response is
    # returned in JSON.
    response = chat(
        model="gemma4:31b-cloud",
        messages=messages,
        format="json",
        options={
            'temperature': 0.0,  # 0-1. Lower temperature makes the output more focused and consistent
            'repeat_penalty': 1.0,  # Setting to 1.0 turns OFF the penalty that stops it from repeating itself.
            # >1.0 increases the penalty for repetition, making repetition less likely.
            'num_predict': 25000,  # Give the model plenty of room to write a long answer
            'seed': 42,  # This combines with temperature to produce the EXACT same answer every single time.
            'num_ctx': 32768,  # Add this to handle the image + markdown
        }
    )

    # Log the VLM response to keep a record of what was returned by the variant_image_extraction function.
    logger.info(
        f"variant_image_extraction() response for {os.path.basename(markdown)}:\n{response['message']['content']}")

    # The path to the markdown file, the file size of the markdown file, the number of tokens it cost to query and
    # receive a response from the VLM are all recorded in the 'ollama_token_costs.txt' file, in case the token cost
    # needs to be evaluated.
    with open("ollama_token_costs.txt", 'a') as f:
        f.write(f"{str(datetime.datetime.now())}\n"
                f"Model: gemma4:31b-cloud\n"
                f"variant_image_extraction()\n"
                f"Path to image file : {image}\n"
                f"Size of image file: {os.path.getsize(image)} bytes\n"
                f"Query costed: {response['prompt_eval_count']} tokens\n"
                f"Response costed: {response['eval_count']} tokens\n"
                f"Total cost: {response['prompt_eval_count'] + response['eval_count']} tokens\n\n")

    # Log the number of tokens it cost to generate the response.
    logger.info(f"Query costed: {response['prompt_eval_count']} tokens")
    logger.info(f"Response costed: {response['eval_count']} tokens")
    logger.info(f"Total cost: {response['prompt_eval_count'] + response['eval_count']} tokens")

    # The JSON response is returned.
    return response["message"]["content"]


def variant_image_description(variant, image, markdown, caption=None):

    markdown_content = ""
    with open(markdown, "r") as f:
        markdown_content = f.read()

    LLM_query = (f'Please analyse {variant} using only the attached image and caption for context. Return the outcome '
                 'from the analysis as a JSON object in the following format:'
                 '{'
                 '  "gene": "",'
                 '  "exon": "",'
                 '  "genomic_variant": "",'
                 '  "transcript_variant": "",'
                 '  "protein_variant": "",'
                 '  "protein_domain": "",'
                 '  "inclusion": "",'
                 '  "laboratory": "",'
                 '  "patient_symptoms": "",'
                 '  "clinical_information": "",'
                 '  "clinical_significance": "",'
                 '  "evidence": "",'
                 '  "frequency": "",'
                 '  "het_carriers": "",'
                 '  "hom_carriers": "",'
                 '  "affected_carriers": "",'
                 '  "unaffected_carriers": "",'
                 '  "number_of_meioses": "",'
                 '  "n_mentions": "",'
                 '} '
                 'Populate the JSON using the following instructions:')

    prompt = f"""
    ---------------------
    CONTEXT:
    Caption: {caption}
    Markdown file: {markdown_content}
    ---------------------
    QUESTION: {LLM_query}
    ---------------------
    INSTRUCTIONS:
    - Only analyse {variant}
    - The variant should be analysed in respect of the image.
    - Use all of the content of the image.
    - Extract as much information from the image and markdown file as possible.
    - Extract as much information from the caption as possible, if a caption was provided.
    - Never hallucinate.
    - Never speculate.
    - Only use information from the caption and markdown file for context.
    - Only use information from the markdown file to help populate the JSON if information cannot be provided by the image and caption first. 
    - In the caption and markdown file, match lines that end abruptly to other lines in the same section that start abruptly and check that they make sense.
    - To string fields in the JSON, enter the text "null" in the string field if information cannot be provided by the image or image legend or image caption or table legend or table caption or figure legend or figure caption.
    - To a list field in the JSON that cannot be populated, return [].
    - Output RAW JSON only. Do not include any introductory text, markdown code blocks (```), or concluding remarks.
    - If you see ' &gt; ' in the text, replace it with '>'. Make sure the whitespaces either side of &gt; are removed.
    - Convert special markdown syntax into plain text characters.
    - Never miss a variant.
    - Analyse the entire variant.
    - Never omit keys in the JSON. 
    - Never hallucinate.
    - Never speculate.
    - If you are unable to extract any clinically relevant information or cannot understand the information, let the user know.  
    - To the 'gene' key, assign the gene symbol of the gene that the corresponding variant is in.
    - To the 'genomic_variant' key, assign the genomic variant as it appears in the image or caption or markdown file. It should start with 'g.' but might not. Remove any whitespace in the variant nomenclature.
    - To the 'transcript_variant' key, assign the variant described at the transcript level as it appears in the image or caption or markdown file. It should start with 'c.' but might not. Remove any whitespace in the variant nomenclature.
    - To the 'exon' key, assign the exon of the gene that the corresponding variant is in. 
    - To the 'protein_variant' key, assign the variant described at the protein level as it appears in the image or caption or markdown file. It should start with 'p.' but might not. Remove any whitespace in the variant nomenclature.
    - To the 'protein_domain' key, assign the protein domain of the protein that the corresponding variant is in.
    - To the 'inclusion' key, assign a detailed description of how the variant was included in the image or caption, including how patients were recruited into the study and the method by which the variant was identified, and any other information that is relevant. Include a description of the precise characteristics that were used to define clinical FH in patients.
    - To the 'laboratory' key, assign the laboratory where the variant was identified, where the patient was tested and the country that the laboratories are in.
    - To the 'patient_information' key, assign the clinical information, symptoms and country of origin of each patient carrying the variant mentioned in the image.
    - To the 'clinical_information' key, provide and assign a comprehensive and profound analysis and description of all of the the clinical information related to {variant} from the image and caption.
    - To the 'clinical_significance' key, provide and assign a comprehensive analysis and description of the clinical significance of {variant} with in the wider context of the image and the caption.
    - To the 'evidence' key, assign a comprehensive and profound analysis and description of the clinical information and significance of the image. Describe the image with intricate and accurate detail as if describing it to someone who cannot see the image. Use the text from the caption and content from the markdown file to support the explanation. Quote text from the caption that was used to substantiate the clinical information and significance, word for word, and convert any markdown syntax in the text into plain text characters.
    - To the 'frequency' key, assign the frequency of {variant} as reported in the image or caption.
    - To the 'het_carriers' key, assign the number of affected carriers with {variant} in a heterozygous genotype reported in the image or caption.
    - To the 'hom_carriers' key, assign the number of affected carriers with {variant} in a homozygous genotype reported in the image or caption.
    - To the 'affected_carriers' key, assign the total number of affected carriers with {variant} reported in the image or caption.
    - To the 'unaffected_carriers' key, assign the total number of unaffected carriers with {variant} reported in the image or caption.
    - One meiosis is the inheritance of a variant from one affected individual to their affected child.
    - To the 'number_of_meioses' key, assign the total number of meioses of {variant} reported in the image or caption.
    - To the 'n_mentions' key, assign the total number of times the variant was mentioned in the markdown file, in any way, whether it be in its genomic, transcript or protein description or in an abbreviated format.
    """

    messages = [
        {
            "role": "system",
            "content": "You are a genomic clinical scientist with the ability to understand, analyse and interpret the "
                       "content of research papers, articles, and literature stored in markdown format. Use only "
                       "content from the markdown files for context. Do not hallucinate. Do not speculate."
        },
        {
            "role": "user",
            "content": f"{prompt}",
            "images": [image]
        }
    ]

    # The chat() function sends the above message to a specified model with additional parameters. The response is
    # returned in JSON.
    response = chat(
        model = "deepseek-v4.1-flash:1017-cloud",
        messages = messages,
        format="json",
        options={
            'temperature': 0.0,     # 0-1. Lower temperature makes the output more focused and consistent
            'repeat_penalty': 1.0,  # Setting to 1.0 turns OFF the penalty that stops it from repeating itself.
                                    # >1.0 increases the penalty for repetition, making repetition less likely.
            'num_predict': 25000,   # Give the model plenty of room to write a long answer
            'seed': 42,             # This combines with temperature to produce the EXACT same answer every single time.
            'num_ctx': 1000000,       # Add this to handle the image + markdown
        }
    )

    # Log the VLM response to keep a record of what was returned by the variant_image_description function.
    logger.info(f"variant_image_description() response for {os.path.basename(markdown)}:\n{response['message']['content']}")

    # The path to the markdown file, the file size of the markdown file, the number of tokens it cost to query and
    # receive a response from the VLM are all recorded in the 'ollama_token_costs.txt' file, in case the token cost
    # needs to be evaluated.
    with open("ollama_token_costs.txt", 'a') as f:
        f.write(f"{str(datetime.datetime.now())}\n"
                f"Model: deepseek-v4.1-flash:1017-cloud\n"
                f"variant_image_description()\n"
                f"Path to image file: {image}\n"
                f"Size of image file: {os.path.getsize(image)} bytes\n"
                f"Query costed: {response['prompt_eval_count']} tokens\n"
                f"Response costed: {response['eval_count']} tokens\n"
                f"Total cost: {response['prompt_eval_count'] + response['eval_count']} tokens\n\n")

    # Log the number of tokens it cost to generate the response.
    logger.info(f"Query costed: {response['prompt_eval_count']} tokens")
    logger.info(f"Response costed: {response['eval_count']} tokens")
    logger.info(f"Total cost: {response['prompt_eval_count'] + response['eval_count']} tokens")

    # The JSON response is returned.
    return response["message"]["content"]