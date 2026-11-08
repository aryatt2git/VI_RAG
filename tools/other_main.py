import os
import shutil
import json
import time
import copy
from datetime import timedelta
from sys import exc_info

from docling_functions import *
from chunk_text import chunkText
from vlm_functions import *
from LLM_functions import *
from qdrant_functions import qdrantImport
from qdrant_client import QdrantClient
from FlagEmbedding import BGEM3FlagModel
from logger import setup_logging
import logging

# Record the time that importation was started.
start_time = time.perf_counter()

# Setup logging.
setup_logging()
logger = logging.getLogger(__name__)

# pdf_path = '/Users/arjun/PycharmProjects/VI_RAG/VI_RAG/tools/Hori et al 2019 PMID 31491741.pdf'

# Log that the embedding model is loading.
logger.info("Loading BAAI/bge-m3 model.")

# The BGE-M3 embedding model is loaded to embed text into up to 1024 tokens.
model = BGEM3FlagModel('BAAI/bge-m3')  # , use_fp16=True
# chunkText is a function used to split text into chunks of 976 tokens.
splitter = chunkText()

# Temporary filepath to References directory where the .pdf files should be kept.
References = '/Users/arjun/PycharmProjects/VI_RAG/VI_RAG/tools/References/'
images_dir = '/Users/arjun/PycharmProjects/VI_RAG/VI_RAG/tools/images'

ref_pmid_ls = []
# Iterate through each file in every directory and subdirectory in the 'References/' folder.
for file in os.listdir(References):

    re_match = re.search(r'PMID\s(\d+)', file)

    if (re_match is not None):
        pmid = re_match.group(1).strip()

        if pmid not in ref_pmid_ls:
            ref_pmid_ls.append(pmid)

for pmid in ref_pmid_ls:

    if pmid != "25741862":
        continue

    pmid_filepaths = []
    markdown_filepaths = []
    images_filepaths = []

    for file in os.listdir(References):
        if pmid in file:
            pmid_filepaths.append(os.path.join(os.path.abspath("References"), file))

    variant_dicts = []
    title = ""
    authors = []

    for filepath in pmid_filepaths:
        #markdown = docling_PDF2Text(filepath)
        docling_ImageExport(filepath)

        basename = os.path.splitext(os.path.basename(filepath))[0]
        pdf_name = basename.strip().replace(" ", "_")
        markdown = f'{pdf_name}.md'

        """
        response = variant_text_extraction(markdown)
        clean_response = response.replace("```json", "").replace("```", "")

        data = json.loads(clean_response)
        print(json.dumps(data, indent=4))

        variants = data['variants']
        for variant in variants:

            response = variant_text_description(variant, markdown)
            clean_response = response.replace("```json", "").replace("```", "")
            data = json.loads(clean_response)

            print(json.dumps(data, indent=4))

            variant_dicts.append(data)
        """

        for image in os.listdir(images_dir):

            image_path = os.path.join(images_dir,image)

            clinical_importance = imageChecker(image_path)

            if clinical_importance == "False":
                logger.info(f"Image is not clinically significant. Skipping image: {image}")
                with open("image_files.txt", "a") as f:
                    f.write(f"{image}\n")
                continue

            response = variant_image_extraction(image_path, markdown)
            clean_response = response.replace("```json", "").replace("```python", "").replace("```", "")

            data = json.loads(clean_response)

            print(json.dumps(data, indent=4))

            caption = ""
            if data['caption'] != "null":
                caption = data['caption']

            for variant in data['variants']:

                response = variant_image_description(variant, image_path, markdown, caption)
                clean_response = response.replace("```json", "").replace("```", "")
                data = json.loads(clean_response)

                print(json.dumps(data, indent=4))

                variant_dicts.append(data)

            os.remove(image_path)





"""
        markdown_filepaths.append(markdown_filepath)

        images = docling_ImageExport(filepath)

        for image in os.listdir(images_dir):

            if image is None:
                continue

            check_resp = imageChecker(os.path.join(images_dir, image))

            # If the VLM returns False, the image is skipped and added to 'image_files.txt' so that tokens
            # aren't spent on checking the image again in the future.
            if check_resp == "False":
                logger.info(f"Image is not clinically significant. Skipping image: {image}")
                with open("image_files.txt", "a") as f:
                    f.write(f"{image}\n")
                continue

            response = vlm_ImageDescription_two(pmid, os.path.join(images_dir, image), markdown_filepath)
            data = json.loads(response)
            print(json.dumps(data, indent=4))
"""