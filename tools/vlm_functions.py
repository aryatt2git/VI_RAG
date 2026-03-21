from ollama import generate, chat

def vlm_loadContext(markdown_path):

    with open(markdown_path, 'r', encoding="utf-8") as f:
        md_text = f.read()

    messages = [
        {
            "role": "system",
            "content": "You are the best text parser and genomic clinical scientist ever because of your ability to "
                       "understand the content and structure of research papers/articles/literature stored in markdown "
                       "files."
        },
        {
            "role": "user",
            "content": f"I am going to provide a markdown file to be used as context. Please recognise and memorise every detail in it. Please commit as much as information to memory as possible. Acknowledge with 'Markdown Loaded'.\nContext:\n{md_text})"
        }
    ]

    response = chat(
        model = "qwen3.5:397b-cloud",
        messages = messages,
        options = {"num_ctx": 32768},
        keep_alive = "60m"
    )

    print("---Loading context to memory---")

    #print(response["message"])

    with open("ollama_token_costs.txt", 'a') as f:
        f.write(f"vlm_loadContext: {markdown_path}\n"
                f"Query costed: {response['prompt_eval_count']} tokens\n"
                f"Response costed: {response['eval_count']} tokens\n"
                f"Total cost: {response['prompt_eval_count'] + response['eval_count']} tokens\n")

    print(f"Query costed: {response['prompt_eval_count']} tokens")
    print(f"Response costed: {response['eval_count']} tokens")
    print(f"Total cost: {response['prompt_eval_count'] + response['eval_count']} tokens")

    messages.append(response["message"])

    return messages


def vlm_TextExtraction(context, markdown_path):

    VLM_query = (
        'Process the markdown file in accordance with the instructions below. Return each subsection of every '
        'section/subsection/subsection of a subsection from the markdown file in accordance with the following JSON '
        'format:'
        '{"title": "", "authors": [], "section_header": "", "subsection_header": "", "sub_subsection_header": "", "text": "", "genes_mentioned": [], variant_count": "", "variants": [{"gene": "", "genomic_variant": "", "transcript_variant": "", "exon": "", "protein_variant": "", "protein_domain": "", "clinical_information": "", "clinical_significance": "", "frequency": "", "het_carriers": "", "hom_carriers": "", "affected_carriers": "", "unaffected_carriers": ""}]}'
    )

    prompt = f"""
    ---------------------
    CONTEXT: {list(context)}
    ---------------------
    QUESTION: {VLM_query}
    ---------------------
    INSTRUCTIONS: 
    - Extract as much information from the markdown file as possible.
    - Use all of the content from the markdown file I sent to you earlier for context.
    - Only use information from the markdown file for context.
    - Split the text in the markdown file into sections, subsections, and subsections of subsections, if necessary.
    - The Abstract should be considered a whole section without any subsections or subsections of subsections
    - Remove all images/table/figure legends/captions/annotations from the markdown file content.
    - Remove all images, figures and and tables from the markdown file content.
    - Remove all in-line references to the articles listed at the end of the markdown file.
    - Remove all references listed after the references header. Also remove the references header.
    - Remove the acknowledgement section and acknowledgement header.
    - Remove the contributions section and contribution header.
    - Remove all sections that have no clinical relevance.
    - Match lines that end abruptly to other lines in the same section that start abruptly and check that they make sense.
    - Before the methods section, remove all text that is not a part of the abstract and introduction sections.
    - If you see ' &gt; ' in the text, replace it with '>'. Mark sure the whitespaces either side of &gt; are removed.
    - Replace special markdown syntax with the original characters.
    - Create a list of JSON objects for each section/subsection/subsections of subsections, where appropriate.
    - To string fields, enter the text "null" in the string field if information cannot be provided by content from the attached markdown file.
    - To a list field that cannot be populated, return [].
    - Return RAW JSON only. Do not include any introductory text, markdown code blocks (```), or concluding remarks.
    - Never miss a paragraph.
    - Analyse each section of the markdown file that I sent to you earlier as context.
    - Never omit keys in the JSON. 
    - Never hallucinate.
    - Never speculate.
    - Do your best.
    - If you are unable to extract any clinically relevant information or cannot understand the information, let the user know.
    - To the 'title' key, assign the name of research paper/article.
    - To the 'authors' key, assign a list of the names of the authors that wrote the research paper/article.
    - To the 'section_header' key, assign the header of the section that the text comes from.
    - To the 'subsection_header' key, assign the subsection header of the section that the text derives from.
    - To the 'sub_subsection_header' key, assign the header of the subsection of the subsection that the text derives from.
    - To the 'text' key, assign the text from the section/subsection/subsection of the subsection specified in this JSON object.
    - To the 'genes_mentioned' key, assign a list of the names of the genes mentioned in the corresponding input text.
    - To the 'variant_count' key, assign the number of variants that appear in the input text.
    - Do not count the genomic and proteomic variant descriptions of the same variant more than once.
    - To the 'variants' key, assign a list of objects for each and every variant mentioned in the corresponding input text.
    - Do not miss any variants out. 
    - To the 'gene' key, assign the gene symbol of the gene that the corresponding variant is in. Use the markdown file that I sent earlier for context.
    - To the 'genomic_variant' key, assign the genomic variant as it appears in the input text. It should start with 'g.' but might not. Remove any whitespace in the variant nomenclature.
    - To the 'transcript_variant' key, assign the variant described at the transcript level as it appears in the corresponding text. It should start with 'c.' but might not.  Remove any whitespace in the variant nomenclature.
    - To the 'exon' key, assign the exon of the gene that the corresponding variant is in. Use the markdown file that I sent earlier for context.
    - To the 'protein_variant' key, assign the variant described at the protein level as it appears in the corresponding text. It should start with 'p.' but might not. Remove any whitespace in the variant nomenclature.
    - To the 'protein_domain' key, assign the protein domain of the protein that the corresponding variant is in. Use the markdown file that I sent earlier for context.
    - To the 'clinical_information' key, provide and assign a comprehensive analysis and description of the clinical information related to the corresponding variant from the corresponding text. Use the markdown file that I sent earlier for context.
    - To the 'clinical_significance' key, provide and assign a comprehensive analysis and description of the clinical significance of the corresponding variant from the the corresponding text. Use the markdown file that I sent earlier for context.
    - To the 'frequency' key, assign the frequency of the corresponding variant as described in corresponding text.
    - To the 'het_carriers' key, assign the number of affected carriers with the corresponding variant in a heterozygous genotype counted in the corresponding text.
    - To the 'hom_carriers' key, assign the number of affected carriers with the corresponding variant in a homozygous genotype counted in the corresponding text.
    - To the 'affected_carriers' key, assign the total number of affected carriers with the corresponding variant counted in the corresponding text.
    - To the 'unaffected_carriers' key, assign the total number of unaffected carriers with the corresponding variant counted in the corresponding text.
    """
    messages = [
        {
            "role": "system",
            "content": "You are the best text parser and genomic clinical scientist ever because of your ability to "
                       "understand the content and structure of research papers/articles/literature stored in markdown "
                       "files. Include only text from the markdown file I sent to you before for context. Do not "
                       "hallucinate. Do not speculate."
        },
        {
            "role": "user",
            "content": f"{prompt}"
        }
    ]

    response = chat(
        model = 'qwen3.5:397b-cloud',
        messages = messages,
        format="json",
        options={
            'temperature': 0.0,  # Lower temperature makes the output more focused and consistent
            'repeat_penalty': 1.0,  # Setting to 1.0 turns OFF the penalty that stops it from repeating itself
            'num_predict': 25000,  # Give the model plenty of room to write a long answer
            'seed': 42,  # Uncomment this if you want the EXACT same answer every single time
            'num_ctx': 32768,  # Add this to handle the image + markdown
        },
        keep_alive = 0  # This forces a fresh start for the next run
    )

    print("--- VLM RESPONSE ---")

    print(response["message"]["content"])

    with open("ollama_token_costs.txt", 'a') as f:
        f.write(f"vlm_TextExtraction: {markdown_path}\n"
                f"Query costed: {response['prompt_eval_count']} tokens\n"
                f"Response costed: {response['eval_count']} tokens\n"
                f"Total cost: {response['prompt_eval_count'] + response['eval_count']} tokens\n")

    print(f"Query costed: {response['prompt_eval_count']} tokens")
    print(f"Response costed: {response['eval_count']} tokens")
    print(f"Total cost: {response['prompt_eval_count'] + response['eval_count']} tokens")

    return response["message"]["content"]


def vlm_TextDescription(input_text, context, markdown_path):

    VLM_query = (
        'Understand, analyse and interpret the following text using the the markdown file that I sent to you earlier '
        'for context: '
        f'{input_text}'
        'Do not hallucinate. Do not speculate. Process the input text in accordance with the instructions. Return '
        'answer in accordance with the following JSON format:'
        '{"description": ""}'
    )

    prompt = f"""
    ---------------------
    CONTEXT: {list(context)}
    ---------------------
    QUESTION: {VLM_query}
    ---------------------
    INSTRUCTIONS: 
    - To string fields, enter the text "null" in the string field if information cannot be provided by content from the attached markdown file.
    - Return RAW JSON only. Do not include any introductory text, markdown code blocks (```), or concluding remarks.
    - Only use information from the markdown file that I sent earlier to provide context to the text.
    - Never miss any text.
    - Analyse all of the input text.
    - Remove any whitespace from every variant nomenclature that appears in the text.
    - Never omit keys in the JSON. 
    - Never hallucinate.
    - Never speculate.
    - Do your best.
    - To the 'description' key, provide and assign a comprehensive analysis and description of the clinical information from the input text using the markdown file that I sent earlier for context. The input_text is the focus of the description.
    """

    messages = [
        {
            "role": "system",
            "content": "You are the best genomic clinical scientist ever because of your ability to understand, "
                       "analyse and interpret the content of research papers/articles/literature stored in markdown "
                       "format. Use only the input text and the markdown file context I sent to you earlier for "
                       "context. Do not hallucinate. Do not speculate."
        },
        {
            "role": "user",
            "content": f"{prompt}"
        }
    ]

    response = chat(
        model='qwen3.5:397b-cloud',
        messages = messages,
        format="json",
        options={
            'temperature': 0.0,  # Lower temperature makes the output more focused and consistent
            'repeat_penalty': 1.0,  # Setting to 1.0 turns OFF the penalty that stops it from repeating itself
            'num_predict': 25000,  # Give the model plenty of room to write a long answer
            'seed': 42,  # Uncomment this if you want the EXACT same answer every single time
            'num_ctx': 32768,  # Add this to handle the image + markdown
        },
        keep_alive = 0  # This forces a fresh start for the next run
    )

    print(f"------VLM EXTRACTING TEXT FROM ------")

    #print(response["message"]["content"])

    with open("ollama_token_costs.txt", 'a') as f:
        f.write(f"vlm_TextDescription: {markdown_path}\n"
                f"text={input_text}\n"
                f"Query costed: {response['prompt_eval_count']} tokens\n"
                f"Response costed: {response['eval_count']} tokens\n"
                f"Total cost: {response['prompt_eval_count'] + response['eval_count']} tokens\n")

    print(f"Query costed: {response['prompt_eval_count']} tokens")
    print(f"Response costed: {response['eval_count']} tokens")
    print(f"Total cost: {response['prompt_eval_count'] + response['eval_count']} tokens")

    return response["message"]["content"]


def vlm_ImageDescription(image_path, context):

    VLM_query = (
        'You are the best text parser and genomic clinical scientist ever because of your ability to understand the '
        'content and structure of research papers/articles/literature stored in markdown format. Include only text from '
        'the markdown file I sent to you before for context. Do not hallucinate. Do not speculate.'
        'Process the image using the markdown file as context in accordance with the instructions below. Analyse the '
        'image as comprehensively as possible. Do not omit any clinically relevant information. '
        'Structure your output into the following JSON format:'
        '{"title": "", "authors": [], "type": "", "figure_table_no": "", "sub_figure_table_no": "", "caption": "", "description": "", "table": "", "genes_mentioned": [], "variant_count": "", "variants": [{"gene": "", "genomic_variant": "", "transcript_variant": "", "exon": "", "protein_variant": "", "protein_domain": "", "clinical_information": "", "clinical_significance": "", "evidence": "", "frequency": "", "het_carriers": "", "hom_carriers": "", "affected_carriers": "", "unaffected_carriers": ""}]}'
    )

    prompt = f"""
    ---------------------
    CONTEXT: {list(context)}
    ---------------------
    QUESTION: {VLM_query}
    ---------------------
    INSTRUCTIONS: 
    - Extract as much information from the attached image as possible.
    - Only use information from the attached image and the markdown file that I sent to you earlier for context.
    - Only analyse images that are referenced inside table/figure legends/captions in the markdown file that I sent to you earlier for context. Other images should be skipped.
    - Match the images to table/figure legends/captions in the markdown file that I sent to you earlier for context.
    - Know that the image's filename does not necessarily correspond with the image number, figure number or table number in the markdown file. Instead, compare the information from the image to the information in the figure/image/table legends/captions to find which legend/caption describes the image most accurately.
    - To string fields, enter the text "null" in the string field if information cannot be provided by the image/figure/table legend/caption.
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
    - To the 'figure_table_no' key, assign the table or figure number in the caption/legend matched with the image from the markdown file that I sent to you earlier provided as context.
    - To the 'sub_figure_table_no' key, assign the table or figure number in the caption/legend matched with the subsection of the image from the markdown file that I sent to you earlier provided as context. For example, 'B' would be assigned to this key for Figure 2B.
    - To the 'caption' key, assign the text provided in the caption/legend matched with the image from the markdown file that I sent to you earlier provided as context.
    - To the 'description' key, provide and assign a comprehensive and profound analysis and description of the clinical information and significance of the image. Describe the image as if speaking to someone who cannot see the image. Use the text from the caption/legend matched with the image to support the explanation.
    - To the 'table' key, if the image is of a table, assign a textualised version of the table that can be embedded by a text embedder.
    - To the 'genes_mentioned' key, assign a list of the name of the genes mentioned in either the image or the caption/legend matched with the image from the markdown file that I sent to you earlier provided as context.
    - To the 'variant_count' key, assign the number of variants that appear in the image. Do not provide any more information that the total number of variants that appear in the image. Do not include any other characters to describe the number than integers.
    - To the 'variants' key, assign a list of objects for each and every variant mentioned in either the image or the caption/legend matched with the image from the markdown file provided as context. Do not miss any variants out. 
    - To the 'gene' key, assign the gene symbol of the gene that the corresponding variant is in.
    - To the 'genomic_variant' key, assign the genomic variant as it appears in the image or the caption/legend matched with the image from the markdown file provided as context. It should start with 'g.' but might not. Remove any whitespace in the variant nomenclature.
    - To the 'transcript_variant' key, assign the variant described at the transcript level as it appears in the image or the caption/legend matched with the image from the markdown file provided as context. It should start with 'c.' but might not. Remove any whitespace in the variant nomenclature.
    - To the 'exon' key, assign the exon of the gene that the corresponding variant is in.
    - To the 'protein_variant' key, assign the variant described at the protein level as it appears in the image or the caption/legend matched with the image from the markdown file provided as context. It should start with 'p.' but might not. Remove any whitespace in the variant nomenclature.
    - To the 'protein_domain' key, assign the protein domain of the protein that the corresponding variant is in.
    - To the 'clinical_information' key, provide and assign a comprehensive and profound analysis and description of the clinical information related to the corresponding variant from the image or the caption/legend matched with the image from the markdown file provided as context.
    - To the 'clinical_significance' key, provide and assign a comprehensive analysis and description of the clinical significance of the corresponding variant from the image or the caption/legend matched with the image from the markdown file provided as context.
    - To the 'evidence' key, assign the text from the image or the caption/legend matched with the image from the markdown file provided as context that was used to substantiate the clinical information and significance.
    - To the 'frequency' key, assign the frequency of the corresponding variant as described in the image or the caption/legend matched with the image from the markdown file provided as context.
    - To the 'het_carriers' key, assign the number of affected carriers with the corresponding variant in a heterozygous genotype counted in the image or the caption/legend matched with the image from the markdown file provided as context.
    - To the 'hom_carriers' key, assign the number of affected carriers with the corresponding variant in a homozygous genotype counted in the image or the caption/legend matched with the image from the markdown file provided as context.
    - To the 'affected_carriers' key, assign the total number of affected carriers with the corresponding variant counted in the image or the caption/legend matched with the image from the markdown file provided as context.
    - To the 'unaffected_carriers' key, assign the total number of unaffected carriers with the corresponding variant counted in the image or the caption/legend matched with the image from the markdown file provided as context.
    """

    messages = [
        {
            "role": "system",
            "content": "You are the best genomic clinical scientist ever because of your ability to understand, "
                       "analyse and interpret the content of research papers/articles/literature stored in markdown "
                       "format. Use only the input text and the markdown file context I sent to you earlier for "
                       "context. Do not hallucinate. Do not speculate."
        },
        {
            "role": "user",
            "content": f"{prompt}",
            "images": [image_path]
        }
    ]

    response = chat(
        model="qwen3.5:397b-cloud",
        messages = messages,
        format="json",
        options={
            'temperature': 0.0,  # Lower temperature makes the output more focused and consistent
            'repeat_penalty': 1.0,  # Setting to 1.0 turns OFF the penalty that stops it from repeating itself
            'num_predict': 25000,  # Give the model plenty of room to write a long answer
            'seed': 42,  # Uncomment this if you want the EXACT same answer every single time
            'num_ctx': 32768,  # Add this to handle the image + markdown
        },
        keep_alive="20m"  # This forces a fresh start for the next run
    )

    print("--- VLM RESPONSE ---")

    print(response["message"]["content"])

    with open("ollama_token_costs.txt", 'a') as f:
        f.write(f"vlm_ImageDescription: {image_path}\n"
                f"Query costed: {response['prompt_eval_count']} tokens\n"
                f"Response costed: {response['eval_count']} tokens\n"
                f"Total cost: {response['prompt_eval_count'] + response['eval_count']} tokens\n")

    print(f"Query costed: {response['prompt_eval_count']} tokens")
    print(f"Response costed: {response['eval_count']} tokens")
    print(f"Total cost: {response['prompt_eval_count'] + response['eval_count']} tokens")

    return response["message"]["content"]