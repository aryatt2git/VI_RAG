"""
This is the main script for importing .pdf files into the Qdrant vector database.
It combines functions from other scripts to convert .pdf files into markdown files.
Each section from the .pdf file is chunked by token size and converted into a pyhon dictionary.
The python dictionary contains additional meta-data from the .pdf to describe the important information in the chunk.
"""

import os
import shutil
import json
import time
import copy
from docling_functions import *
from chunk_text import chunkText
from vlm_functions import *
from qdrant_functions import qdrantImport
from qdrant_client import QdrantClient
from FlagEmbedding import BGEM3FlagModel

#pdf_path = '/Users/arjun/PycharmProjects/VI_RAG/VI_RAG/tools/Hori et al 2019 PMID 31491741.pdf'


print("---Loading model")

# The BGE-M3 embedding model is loaded to embed text into up to 1024 tokens.
model = BGEM3FlagModel('BAAI/bge-m3') #, use_fp16=True
# chunkText is a function used to split text into chunks of 976 tokens.
splitter = chunkText()

# Temporary filepath to References directory where the .pdf files should be kept.
References = '/Users/arjun/PycharmProjects/VI_RAG/VI_RAG/tools/References/'

# Iterate through each file in every directory and subdirectory in the 'References/' folder.
for root, dir, files in os.walk(References, topdown=False):
    for file in files:
        # Specify only .pdf files.
        if file.endswith('.pdf'):

            # Get the path to the file currently being iterated over.
            pdf_path = os.path.join(os.path.abspath(root), file)

            # Create a list of .pdf filepaths that have been recorded in the 'pdf_paths.txt'. This file is populated
            # at the end of this script, once the .pdf file being iterated over has been successfully imported into the
            # Qdrant vector database. This is used to quickly skip over .pdfs that have already been imported into
            # the vector database.
            pdf_list = []
            if os.path.exists("pdf_paths.txt"):
                with open("pdf_paths.txt", "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        pdf_list.append(line.strip())

            if pdf_path in pdf_list:
                print(f"---Skipping entire import of: {pdf_path}")
                continue

            # When converting the .pdf file into a markdown file, images in the .pdf file are extracted and stored in
            # a separate directory called 'images'. If this file already exists, it is removed so that it can store
            # images from the .pdf file currently being iterated over.
            if os.path.exists("./images/"):
                print(f"---Deleting images directory...")
                shutil.rmtree("./images/")

            # When converting the .pdf file into a markdown file, the markdown file is stored in the 'tools' directory.
            # If a markdown file already exists, it is removed so that the .pdf file currently being converted  into a
            # markdown file is the only markdown file in the folder.
            for md_file in os.listdir("./"):
                if md_file.endswith('.md'):
                    print(f"---Deleting markdown file: {md_file}")
                    os.remove(os.path.join("./" , md_file))

            # Retrieve the .pdf filename and use it as the name for the markdown file.
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            markdown_name = f'{base_name.strip().replace(" ", "_")}.md'

            # Importing a .pdf into the vector database consists of two main steps:
            # 1) Importing the text from each section.
            # 2) Importing text-based descriptions of images from the .pdf.
            # The markdown filename is recorded in 'markdown_files.txt' file once the content of the markdown file has
            # been imported into the vector database. The markdown filenames in 'markdown_files.txt' are assigned to
            # 'markdown_list'. If the name of the markdown file that is due to be generated during this iteration is
            # in the list, then generating and importing this markdown file is skipped. The script then moves to
            # importing images from the .pdf on line 337.
            markdown_list=[]
            if os.path.exists("markdown_files.txt"):
                with open("markdown_files.txt", "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        markdown_list.append(line.strip())

            if markdown_name in markdown_list:
                print(f"---Skipping text import of: {markdown_name}")
                pass

            # If the .pdf markdown file has not been imported into the vector database, a markdown file of the .pdf
            # from the current iteration is generated.
            else:

                print(f"---Importing {file}")

                # The docling_PFD2Text generates a markdown file from the .pdf filepath. The filepath to the markdown
                # file is returned by the docling_PDF2Text function and assigned to the 'markdown_path' variable.
                markdown_path = docling_PDF2Text(pdf_path)
                #print(markdown_path)

                # The markdown file is provided as context to the Visual Language Model (VLM) via the vlm_loadContext
                # function which commits the markdown file to the VLM's 'memory' for 60 minutes.
                # This was done to reduce the cost of tokens sent to Ollama but it did not change the cost very much.
                # CONSIDER REMOVING THIS STEP AND REMEMBER TO CHANGE THE VLM_TEXTEXTRACTION AND THE
                # VLM_IMAGEDESCRIPTION INPUT PARAMETERS AND FUNCTIONS.
                context = vlm_loadContext(markdown_path=markdown_path)

                # Create an empty variable to store the list of python dictionaries in JSON format produced by the VLM.
                ext_json_list = None

                # Make 10 attempts to retrieve a response from the VLM until a response has been retrieved.
                for attempt in range(10):

                    # If ext_json_list has been populated, iterate out of the loop.
                    if ext_json_list:
                        continue

                    # Sometimes the VLM does not produce a response that can be loaded into Python without raising
                    # an exception. Therefore, a simple try and except statement is used to retry parsing a list of
                    # each section from the .pdf into ext_json_list.
                    try:
                        # The vlm_TextExtraction function is used to parse each section in the .pdf file into a JSON
                        # object along with accompanying meta-data. Each section if then store in a list.
                        ext_response = vlm_TextExtraction(context=context, markdown_path=markdown_path)
                        # The list is then loaded into python.
                        ext_json_list = json.loads(ext_response)

                        # If 'ext_json_list' variable is None or the VLM returns an empty list, iterate through
                        # another attempt.
                        if ext_json_list is None:
                            continue
                        elif len(ext_json_list) == 0:
                            continue

                    except:
                        continue

                # If after 10 attempts, 'ext_json_list' variable is None or the VLM returns an empty list, the markdown
                # file is not imported into the vector database.
                if ext_json_list is None:
                    continue
                elif len(ext_json_list) == 0:
                    continue

                # Iterate through each python dictionary in the 'ext_json_list' list.
                for resp_dict in ext_json_list:

                    # Iterate over each key and value in the dictionary.
                    for key, value in resp_dict.items():

                        # If any value is None, assign an empty array  or string or 0 value to the key so the entire
                        # dictionary can still be imported into the vector database.
                        if value is None:
                            if key in ["authors", "genes_mentioned", "variants"]:
                                resp_dict[key] = []
                            elif key in ["title", "section_header", "subsection_header", "sub_subsection_header"]:
                                resp_dict[key] = ""
                            elif key == "variant_count":
                                resp_dict[key] = 0

                        # A list of dictionaries are assigned to the 'variants' key. Each dictionary consists of
                        # information about a variant that was mentioned in the corresponding section of the .pdf.
                        # This code iterates over each dictionary and replaces any NoneType values with an empty string
                        # to ensure that the dictionary can still be imported into the vector database.
                        if key == "variants" and isinstance(value, list):
                            for variant in value:
                                for v_key, v_value in variant.items():
                                    if v_value is None:
                                        variant[v_key] = ""

                # Iterate through each python dictionary in the 'ext_json_list' list.
                for resp_dict in ext_json_list:

                    # The content of each section/subsection (assigned to the 'text' key in the python dictionary) is
                    # split into chunks of text, each of which would consist of 976 tokens, as permitted by the BGE-M3
                    # embedding model.
                    chunks = splitter.split_text(resp_dict['text'])

                    # Iterate over each chunk in order along with its index number (i).
                    for i, chunk in enumerate(chunks):

                        #Create a dictionary for each chunk that will be imported into the Qdrant vector database.
                        qdrant_dict = {}

                        # Iterate through each key and value in the python dictionary that is currently being processed.
                        for key, value in resp_dict.items():

                            # Copy all keys and values from the python dictionary produced by the VLM, except for the
                            # 'text' key and its associated value.
                            if key != 'text':
                                qdrant_dict[key] = copy.deepcopy(value)

                        # The VLM may have assigned "null" to some of the 'variant_count' keys. It would be better to
                        # return the integer 0 so that data imported into the vector database can be easily filtered.
                        if qdrant_dict["variant_count"] == "null":
                            qdrant_dict["variant_count"] = 0

                        # The filepath to the .pdf is also added to qdrant_dict as this might provide extra metadata
                        # that could help with data retrieval.
                        # CONSIDER REMOVING THIS AS THIS MIGHT ALSO PERMIT DUPLICATE PAPERS TO BE ENTERED INTO THE
                        # VECTOR DATABASE IF THE FILEPATH IS DIFFERENT (I.E. DIFFERENT USERS COULD IMPORT THE SAME PAPER
                        # AS ONE THAT ALREADY EXISTS IN THE VECTOR DATABASE BECAUSE THEY WILL HAVE DIFFERENT ABSOLUTE
                        # PATHS FOR THE MARKDOWN FILE).
                        qdrant_dict["path"] = markdown_path

                        # The chunk index is assigned to the 'chunk_idx' key, in the 'qdrant_dict' dictionary.
                        qdrant_dict["chunk_idx"] = i + 1
                        # The chunk text is assigned to the 'chunk' key in the 'qdrant_dict' dictionary.
                        qdrant_dict["chunk"] = chunk

                        # An empty variable is created to store a description of the chunk, generated by the VLM, to
                        # provide greater context of the chunk in relation to rest of the .pdf.
                        description_json = None

                        # Make 5 attempts to retrieve a response from the VLM.
                        for attempt in range(5):

                            # If the 'description_json' variable becomes populated with a python dictionary, with a
                            # 'description' in it, break out of the loop and continue with the remainder of the script.
                            if description_json and "description" in description_json:
                                break

                            # Sometimes the VLM does not produce a response that can be loaded into Python without
                            # raising an exception. Therefore, a simple try and except statement is used to retry
                            # parsing a list of each section from the .pdf into ext_json_list.
                            try:
                                description_resp = vlm_TextDescription(input_text=chunk, context=context, markdown_path=markdown_path)
                                description_json = json.loads(description_resp)

                                # If the description returned by the VLM is a NoneType value or the description key
                                # (and value) are missing, try again.
                                if description_json is None or "description" not in description_json:
                                    continue

                            # If an exception arises, wait twice the current attempt and then try again.
                            except:
                                wait_time = 2**(attempt+1)
                                time.sleep(wait_time)
                                continue

                        # If the description returned by the VLM is a NoneType value or the description key (and value)
                        # are missing, insert a 'description' key with a "null" value.
                        if description_json is None or "description" not in description_json:
                            qdrant_dict["description"] = "null"
                        # Otherwise, insert the description from the VLM response into the dictionary due to be
                        # imported into the vector database.
                        else:
                            qdrant_dict["description"] = description_json["description"]

                        # Create a list of genes mentioned in the chunk of text, to support accurate indexing in the
                        # vector database and easier retrieval.
                        chunk_genes = []
                        if len(qdrant_dict['genes_mentioned']) > 0:
                            for gene in qdrant_dict['genes_mentioned']:
                                if gene in chunk:
                                    chunk_genes.append(gene)
                        # Add the list of genes mentioned in the chunk to the Qdrant dictionary due to be imported into
                        # the vector database.
                        qdrant_dict['genes_mentioned'] = chunk_genes

                        # Create a list of dictionaries of variants mentioned in the chunk of text, to support accurate
                        # indexing in the vector database and easier retrieval.
                        chunk_variants = []

                        # If variants have been mentioned in the corresponding subsection, in the original .pdf,
                        # iterate through each variant that has was mentioned.
                        if len(qdrant_dict['variants']) > 0:
                            for variant in qdrant_dict['variants']:

                                # Create a list to store the transcript, genomic and protein nomenclature from the
                                # variant's python dictionary.
                                nom_list = []
                                c_variant = variant["transcript_variant"]
                                g_variant = variant["genomic_variant"]
                                p_variant = variant["protein_variant"]

                                # Parse the variant nomenclatures and append them into the 'nom_list' list.
                                if c_variant and c_variant != "null":
                                    nom_list.append(c_variant)
                                if g_variant and g_variant != "null":
                                    nom_list.append(g_variant)
                                if p_variant and p_variant != "null":
                                    nom_list.append(p_variant)

                                # Search for every nomenclature for each variant in the chunk of text and append the
                                # variant dictionary to the 'chunk_variants' list any of them are present.
                                # CHECK IF THIS WORKS CORRECTLY FOR PROTEIN NOMENCLATURE DESCRIBED AS E.G. A302R IN THE
                                # CHUNK AND p.(Ala302Arg) IN THE DICTIONARY.
                                if len(nom_list) > 0:
                                    for nom in nom_list:
                                        if nom in chunk:
                                            chunk_variants.append(variant)
                                            break
                                else:
                                    continue

                        # Add the list of variant dictionaries to the Qdrant dictionary due to be imported into the
                        # vector database.
                        qdrant_dict['variants'] = chunk_variants

                        # Assign the number of variants mentioned in the chunk to the 'variant_count' key, in the
                        # Qdrant dictionary due to be imported into the vector database.
                        qdrant_dict['variant_count'] = len(qdrant_dict['variants'])

                        # Log the Qdrant dictionary due to be imported into the vector database.
                        print(json.dumps(qdrant_dict, indent=4))

                        # Open the 'chunk_sizes.txt' file to store information about the chunk about to be imported
                        # into the vector database.
                        with open("chunk_sizes.txt", "a") as f:
                            f.write(f"Text: {markdown_name}\n"
                                    f"Chunk No.: {i+1}\n"
                                    f"Chunk length: {len(chunk)}\n"
                                    f"Dict length: {len(str(qdrant_dict))}\n")

                        # Log the markdown filename being imported into the Qdrant database.
                        print(f"---Loading {markdown_name} into Qdrant database.")

                        # Import the 'qdrant_dict' into the Qdrant vector database.
                        qdrantImport(chunk_dict=qdrant_dict, model=model)

                        # Log that the markdown file was successfully imported into the vector database.
                        print(f"---{markdown_name} successfully loaded into Qdrant database")

                # Add the markdown filename to 'markdown_files.txt', incase the same set of .pdf files are imported
                # into the vector database and markdown files that have already been imported are skipped.
                with open("markdown_files.txt", "a") as f:
                    f.write(f"{markdown_name}\n")

            # docling_FigTableExport function is used to extract images from the .pdf file and return the filepath to
            # the 'images' directory where they are stored.
            images = docling_FigTableExport(pdf_path)

            # Iterate over each image extracted from the .pdf.
            for image in os.listdir(images):

                # Create a list of .pdf image filenames that have been recorded in the 'image_files.txt'. This file is
                # populated at the end of this script, once the image being iterated over has been successfully imported
                # into the Qdrant vector database. This is used to quickly skip over images that have already been
                # imported into the vector database.
                if os.path.exists("image_files.txt"):
                    images_list = []
                    with open("image_files.txt", "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            images_list.append(line.strip())

                    if image in images_list:
                        # Display which image has been skipped to stdout.
                        print(f"Skipping image import of: {image}")
                        continue

                # Specify images that end in the expected file extension (although only images extracted rom the .pdf
                # file should be in the 'images' directory.
                if image.endswith('.png'):

                    # The docling function extracts images from the .pdf and lists them as
                    # <.pdf filename>-picture-{picture_counter}.png. This line of code simply grabs the name of the
                    # .pdf file that the images are from.
                    filename = image.split('-')[0]

                    # Assign the filepath relative to the 'images' directory to the corresponding markdown file for the
                    # same .pdf that the images derive from, to the 'markdown' variable.
                    markdown = os.path.join(images, '..', f'{filename}.md')

                    # Not every image extracted from the .pdf file is clinically relevant. Some images might simply be
                    # a logo or miscellaneous imagery. The imageChecker function uses the VLM to check if the image is
                    # clinically relevant before importing the image into the vector database.
                    check_resp = imageChecker(os.path.join(images, image))

                    # If the VLM returns False, the image is skipped and added to 'image_files.txt' so that tokens
                    # aren't spent on checking the image again.
                    if check_resp == "False":
                        print(f"---Skipping image: {image}")
                        with open("image_files.txt", "a") as f:
                            f.write(f"{image}\n")
                        continue

                    # Display that the image is being prepared for upload in to the vector database.
                    print(f"---Loading: {image}")

                    # Check if the markdown file exists. If it does not, generate it and assign its filepath to the
                    # 'markdown_path' variable.
                    if os.path.exists(markdown):
                        markdown_path = os.path.abspath(os.path.join(images, '..', f'{filename}.md'))
                    else:
                        markdown_path = docling_PDF2Text(pdf_path)

                    # Log the path to the markdown path?
                    #print(markdown_path)

                    # The markdown file is provided as context to the Visual Language Model (VLM) via the
                    # vlm_loadContext function which commits the markdown file to the VLM's 'memory' for 60 minutes.
                    # This was done to reduce the cost of tokens sent to Ollama but it did not change the cost very
                    # much.
                    # CONSIDER REMOVING THIS STEP AND REMEMBER TO CHANGE THE VLM_TEXTEXTRACTION AND THE
                    # VLM_IMAGEDESCRIPTION INPUT PARAMETERS AND FUNCTIONS.
                    context = vlm_loadContext(markdown_path=markdown_path)

                    # Get the absolute path to the image being processed in the current iteration (current image).
                    image_path = os.path.abspath(os.path.join(images, image))
                    print(f"Processing: {image_path}")

                    # The code below this line will makes 5 attempts to extract as many variants as possible from the
                    # current image. Testing showed that the VLM did not extract the same information from a figure,
                    # and it would often miss variants. Therefore, multiple attempts are made to extract as much
                    # information as possible, using the variant count as the determinant, as that is the most
                    # important feature that we want to extract from each image.
                    # The VLM response that is kept, is the one with the highest variant_count, however, not every
                    # image might contain clinically significant information related to a variant. Therefore, the
                    # variant_count starts at -1 so that VLM responses that do not include variants can still be
                    # imported.
                    variant_count = -1

                    # Create an empty 'image_dict' variable that can be populated by the VLM response and be imported
                    # into the Qdrant vector database.
                    image_dict = None

                    # Make 5 attempts to retrieve the 'best' response.
                    for attempt in range(5):
                        # The vlm_ImageDescription function produces a JSON response with a description of the image
                        # and additional metadate that will help to index the image in the vector database.
                        response = vlm_ImageDescription(image_path=image_path, context=context)

                        # Sometimes the VLM does not produce a response that can be loaded into Python without
                        # raising an exception. Therefore, a simple try and except statement is used to retry
                        # creating an appropriate JSON response from the VLM.
                        try:
                            resp_dict = json.loads(response)

                        # If an exception occurs, move on to the next attempt.
                        except:
                            continue

                        # If the JSON returned by the VLM was not loaded into a python dictionary, move on to the next
                        # attempt.
                        if not resp_dict:
                            continue

                        # If the variant_count provided by the VLM is NoneType or "null", assign the integer 0 to the
                        # VLM dictionary key, 'variant_count'.
                        if resp_dict["variant_count"] is None:
                            resp_dict["variant_count"] = 0
                        elif resp_dict["variant_count"] == "null":
                            resp_dict["variant_count"] = 0

                        # Log the variant count.
                        print(f"count = {resp_dict['variant_count']}")

                        # Assign the variant count to the variable 'count'.
                        count = int(resp_dict["variant_count"])

                        # If the value of 'count' (the variant count from the current VLM response) is larger than the
                        # value of 'variant_count', the current VLM response is assigned to the 'image_dict' variable
                        # and replaces the python dictionary that may have already been assigned to it.
                        if count > variant_count:
                            image_dict = resp_dict
                            # If the current VLM response has a higher variant count, then the current VLM's variant
                            # count is assigned to the 'variant_count' variable.
                            variant_count = count
                        # If the value assigned to the 'variant_count' key is less than the value assigned to 'count',
                        # another attempt is made to retrieve a higher value from the VLM.
                        else:
                            continue

                    # The filepath to the current image is also added to the dictionary that is due to be imported into
                    # the Qdrant vector database as this might provide extra metadata that could help with data
                    # retrieval.
                    # CONSIDER REMOVING THIS AS THIS MIGHT ALSO PERMIT DUPLICATE PAPERS TO BE ENTERED INTO THE
                    # VECTOR DATABASE IF THE FILEPATH IS DIFFERENT (I.E. DIFFERENT USERS COULD IMPORT THE SAME PAPER
                    # AS ONE THAT ALREADY EXISTS IN THE VECTOR DATABASE BECAUSE THEY WILL HAVE DIFFERENT ABSOLUTE
                    # PATHS FOR THE MARKDOWN FILE).
                    image_dict['path'] = image_path

                    # Iterate through the keys and values in 'image_dict'.
                    for key, value in image_dict.items():
                        # If a value is a NoneType...
                        if value is None:
                            # ...assign an empty list to any keys that a list was meant to be assigned to.
                            if key in ["authors", "genes_mentioned", "variants"]:
                                image_dict[key] = []
                            # ...assign an empty string to any keys that a string was meant to be assigned to.
                            elif key in ["title", "type", "figure_table_no", "sub_figure_table_no", "text",
                                         "caption", "description", "table", "variant_count"]:
                                image_dict[key] = ""
                            # ...assign an integer of 0 to any keys that a number was meant to be assigned to.
                            elif key == "variant_count":
                                image_dict[key] = 0

                        # Replace any NoneType values in any of the variant dictionaries listed in the 'variants' key,
                        # if a list has been assigned to the 'variants' key.
                        if key == "variants" and isinstance(value, list):
                            for variant in value:
                                for key, value in variant.items():
                                    if value is None:
                                        variant[key] = ""

                    # Log the image dictionary generated by the VLM.
                    print(json.dumps(image_dict, indent=4))

                    # Create the 'chunks' variable to store each chunk from the image dictionary.
                    chunks = None

                    # Figures and tables are distinguished by the value assigned to the 'type' key in the image's
                    # python dictionary.
                    # The VLM is able to convert images of tables into text. Therefore, the image's caption (legend)
                    # from the .pdf markdown file, the VLM's description of the table and the content of the table are
                    # extracted from the image dictionary and combined for chunking.
                    # The combined text is then chunked using the 'chunkText' function assigned to 'splitter'.
                    if image_dict['type'] == 'table':
                        chunks = splitter.split_text(
                            f"Caption: {image_dict['caption']} "
                            f"Description: {image_dict['description']} "
                            f"Table: {image_dict['table']}"
                        )

                    # If the images is of a figure, only the caption and description provided in the image dictionary
                    # by the VLM is combined for chunking.
                    elif image_dict['type'] == 'figure':
                        chunks = splitter.split_text(
                            f"Caption: {image_dict['caption']} "
                            f"Description: {image_dict['description']}"
                        )

                    # Any other image type is chunked similarly to how a figure is chunked.
                    else:
                        chunks = splitter.split_text(
                            f"Caption: {image_dict['caption']} "
                            f"Description: {image_dict['description']}"
                        )

                    # Iterate through each chunk in the 'chunks' variable along with an index.
                    for i, chunk in enumerate(chunks):

                        # Create and empty dictionary to store information about the current chunk in the iteratioh.
                        qdrant_dict = {}

                        # Copy every key and value from the VLM's response into the dictionary that is due to be
                        # imported into the Qdrant vector database.
                        for key, value in image_dict.items():
                            qdrant_dict[key] = copy.deepcopy(value)

                        # Store an index number in the chunk's Qdrant dictionary. 9001 is added to the index number to
                        # ensure that chunks that describe images are indexed and distinguishable from chunks that
                        # derive from the main body of text in the respective .pdf file. This will help us to determine
                        # if evidence derived from an image or text.
                        qdrant_dict["chunk_idx"] = i + 9001

                        # Insert the chunk of text into the dictionary that is due to be uploaded into the Qdrant
                        # vector database.
                        qdrant_dict["chunk"] = chunk

                        # Figures should generate only a single chunk as captions and descriptions should be embedded
                        # within the 976 token size. However, tables can be very long and produce lots of chunks.
                        # Therefore, it is only beneficial to describe the genes and variants that appear in each chunk
                        # from tables.
                        if qdrant_dict['type'] == 'table':
                            # Create an empty list which can be populated by genes that appear in the chunk.
                            chunk_genes = []
                            if len(qdrant_dict['genes_mentioned']) > 0:
                                for gene in qdrant_dict['genes_mentioned']:
                                    if gene in chunk:
                                        chunk_genes.append(gene)

                            # Replace the list of every gene that appears in the table with a list of genes that only
                            # appear in the chunk.
                            qdrant_dict['genes_mentioned'] = chunk_genes

                            # Create an empty list which can be populated by dictionaries of variants that appear in
                            # the chunk, if the VLM recognised variants in the table.
                            chunk_variants = []
                            if len(qdrant_dict['variants']) > 0:

                                # Iterate through each variant dictionary and extract the transcript, proteomic and
                                # genomic descriptions of the respective variant.
                                for variant in qdrant_dict['variants']:

                                    # An empty list is assigned to the 'nom_list' variable so it can hold the variant's
                                    # nomenclatures.
                                    nom_list = []
                                    c_variant = variant["transcript_variant"]
                                    g_variant = variant["genomic_variant"]
                                    p_variant = variant["protein_variant"]

                                    # If the nomenclature is not a NoneType value nor "null", the nomenclature is added
                                    # to the 'nom_list' list.
                                    if c_variant and c_variant != "null":
                                        nom_list.append(c_variant)
                                    if g_variant and g_variant != "null":
                                        nom_list.append(g_variant)
                                    if p_variant and p_variant != "null":
                                        nom_list.append(p_variant)

                                    # If variants have been added to the 'nom_list', search the chunk for any of the
                                    # nomenclatures that have been added to the list. If found, add the respective
                                    # variant's dictionary to the 'chunk_variants' list.
                                    if len(nom_list) > 0:
                                        for nom in nom_list:
                                            if nom in chunk:
                                                chunk_variants.append(variant)
                                                break

                                    # If the nomenclature could not be found in the chunk, move on to the next variant
                                    # dictionary.
                                    else:
                                        continue

                            # Add the list of variant dictionaries assigned to 'chunk_variants' to the 'variants' key
                            # in the dictionary due to be imported into the Qdrant database.
                            qdrant_dict['variants'] = chunk_variants

                            # Change the 'variant_count' value from the number of variants that appear in the
                            # subsection to the number of variants that appear in the chunk.
                            qdrant_dict['variant_count'] = len(qdrant_dict['variants'])

                            # The filepath to the image is also added to qdrant_dict as this might provide extra
                            # metadata that could help with data retrieval.
                            # CONSIDER REMOVING THIS AS THIS MIGHT ALSO PERMIT DUPLICATE PAPERS TO BE ENTERED INTO THE
                            # VECTOR DATABASE IF THE FILEPATH IS DIFFERENT (I.E. DIFFERENT USERS COULD IMPORT THE SAME
                            # PAPER AS ONE THAT ALREADY EXISTS IN THE VECTOR DATABASE BECAUSE THEY WILL HAVE DIFFERENT
                            # ABSOLUTE PATHS FOR THE MARKDOWN FILE).
                            qdrant_dict["path"] = image_path

                        # Log the dictionary of the image that is due to be imported into the Qdrant vector database.
                        print(json.dumps(qdrant_dict, indent=4))

                        # Open the 'chunk_sizes.txt' file to store information about the chunk about to be imported
                        # into the vector database.
                        with open("chunk_sizes.txt", "a") as f:
                            f.write(f"Image: {image}\n"
                                    f"Chunk No.: {i+9001}\n"
                                    f"Chunk length: {len(chunk)}\n"
                                    f"Dict length: {len(str(qdrant_dict))}\n")

                        # Log which image is about to be imported into the vector database.
                        print(f"---Loading {image} into Qdrant database.\n")

                        # Import the 'qdrant_dict' into the Qdrant vector database.
                        qdrantImport(chunk_dict=qdrant_dict, model=model)

                        # Log that the image was successfully imported into the vector database.
                        print(f"---{image} successfully loaded into Qdrant database")

                    # Add the image's filename to 'image_files.txt', so that images that have already been imported are
                    # skipped.
                    with open("image_files.txt", "a") as f:
                        f.write(f"{image}\n")

            # Once the images from the .pdf file have been imported, delete them from the 'images' folder.
            if os.path.exists(images):
                shutil.rmtree(images)

            # Add the .pdf filename to 'pdf_paths.txt', so that .pdf files that have already been imported are skipped.
            with open("pdf_paths.txt", "a") as f:
                f.write(f"{pdf_path}\n")