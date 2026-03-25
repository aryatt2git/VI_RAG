import os
import shutil
import json
import time
import copy

from docling_functions import *
from chunk_text import chunkText
from vlm_functions import *
from weaviate_functions import *
from FlagEmbedding import BGEM3FlagModel

#pdf_path = '/Users/arjun/PycharmProjects/VI_RAG/VI_RAG/tools/Hori et al 2019 PMID 31491741.pdf'

print("---Loading model")
model = BGEM3FlagModel('BAAI/bge-m3') #, use_fp16=True
splitter = chunkText()

References = '/Users/arjun/PycharmProjects/VI_RAG/VI_RAG/tools/References/'

for root, dir, files in os.walk(References, topdown=False):
    for file in files:
        if file.endswith('.pdf'):

            pdf_path = os.path.join(os.path.abspath(root), file)

            pdf_list = []
            if os.path.exists("pdf_paths.txt"):
                with open("pdf_paths.txt", "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        pdf_list.append(line.strip())

            if pdf_path in pdf_list:
                print(f"---Skipping entire import of: {pdf_path}")
                continue

            if os.path.exists("./images/"):
                print(f"---Deleting images directory...")
                shutil.rmtree("./images/")

            for md_file in os.listdir("./"):
                if md_file.endswith('.md'):
                    print(f"---Deleting markdown file: {md_file}")
                    os.remove(os.path.join("./" , md_file))

            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            markdown_name = f'{base_name.strip().replace(" ", "_")}.md'

            markdown_list=[]
            if os.path.exists("markdown_files.txt"):
                with open("markdown_files.txt", "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        markdown_list.append(line.strip())

            if markdown_name in markdown_list:
                print(f"---Skipping text import of: {markdown_name}")
                pass

            else:

                print(f"---Importing {file}")

                markdown_path = docling_PDF2Text(pdf_path)
                #print(markdown_path)

                context = vlm_loadContext(markdown_path=markdown_path)

                chunk_dicts = []
                ext_json_list = None

                for attempt in range(10):
                    if ext_json_list:
                        continue
                    try:
                        ext_response = vlm_TextExtraction(context=context, markdown_path=markdown_path)
                        ext_json_list = json.loads(ext_response)
                    except:
                        continue

                if ext_json_list is None:
                    continue
                elif len(ext_json_list) == 0:
                    continue

                for resp_dict in ext_json_list:

                    for key, value in resp_dict.items():

                        if value is None:
                            if key in ["authors", "genes_mentioned", "variants"]:
                                resp_dict[key] = []
                            elif key in ["title", "section_header", "subsection_header", "sub_subsection_header"]:
                                resp_dict[key] = ""
                            elif key == "variant_count":
                                resp_dict[key] = 0

                        if key == "variants" and isinstance(value, list):
                            for variant in value:
                                for v_key, v_value in variant.items():
                                    if v_value is None:
                                        variant[v_key] = ""

                for resp_dict in ext_json_list:

                    chunks = splitter.split_text(resp_dict['text'])

                    for i, chunk in enumerate(chunks):

                        weaviate_dict = {}

                        for key, value in resp_dict.items():

                            if key != 'text':
                                weaviate_dict[key] = copy.deepcopy(value)

                        if weaviate_dict["variant_count"] == "null":
                            weaviate_dict["variant_count"] = 0

                        weaviate_dict["path"] = markdown_path

                        weaviate_dict["chunk_idx"] = i + 1
                        weaviate_dict["chunk"] = chunk

                        description_json = None
                        for attempt in range(5):
                            if description_json and "description" in description_json:
                                break
                            try:
                                description_resp = vlm_TextDescription(input_text=chunk, context=context, markdown_path=markdown_path)
                                description_json = json.loads(description_resp)
                            except:
                                wait_time = 2**(attempt+1)
                                time.sleep(wait_time)
                                continue

                        if description_json is None or "description" not in description_json:
                            weaviate_dict["description"] = "null"
                        else:
                            weaviate_dict["description"] = description_json["description"]

                        chunk_genes = []
                        if len(weaviate_dict['genes_mentioned']) > 0:
                            for gene in weaviate_dict['genes_mentioned']:
                                if gene in chunk:
                                    chunk_genes.append(gene)

                        weaviate_dict['genes_mentioned'] = chunk_genes

                        chunk_variants = []
                        if len(weaviate_dict['variants']) > 0:
                            for variant in weaviate_dict['variants']:
                                nom_list = []
                                c_variant = variant["transcript_variant"]
                                g_variant = variant["genomic_variant"]
                                p_variant = variant["protein_variant"]

                                if c_variant and c_variant != "null":
                                    nom_list.append(c_variant)
                                if g_variant and g_variant != "null":
                                    nom_list.append(g_variant)
                                if p_variant and p_variant != "null":
                                    nom_list.append(p_variant)

                                if len(nom_list) > 0:
                                    for nom in nom_list:
                                        if nom in chunk:
                                            chunk_variants.append(variant)
                                            break
                                else:
                                    continue

                        weaviate_dict['variants'] = chunk_variants

                        weaviate_dict['variant_count'] = len(weaviate_dict['variants'])

                        print(json.dumps(weaviate_dict, indent=4))

                        with open("chunk_sizes.txt", "a") as f:
                            f.write(f"Text: {markdown_name}\n"
                                    f"Chunk No.: {i+1}\n"
                                    f"Chunk length: {len(chunk)}\n"
                                    f"Dict length: {len(str(weaviate_dict))}\n")

                        chunk_dicts.append(weaviate_dict)

                print(f"---Loading {markdown_name} into weaviate database.\n"
                      f"--- {chunk_dicts} chunks awaiting upload.")
                weaviateImportText(dict_list=chunk_dicts, model=model)
                print(f"---{markdown_name} successfully loaded into weaviate database")

                with open("markdown_files.txt", "a") as f:
                    f.write(f"{markdown_name}\n")


            images = docling_FigTableExport(pdf_path)

            for image in os.listdir(images):

                if os.path.exists("image_files.txt"):
                    images_list = []
                    with open("image_files.txt", "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            images_list.append(line.strip())

                    if image in images_list:
                        print(f"Skipping image import of: {image}")
                        continue

                if image.endswith('.png'):

                    filename = image.split('-')[0]

                    markdown = os.path.join(images, '..', f'{filename}.md')

                    check_resp = imageChecker(os.path.join(images, image))
                    if check_resp == "False":
                        print(f"---Skipping image: {image}")
                        with open("image_files.txt", "a") as f:
                            f.write(f"{image}\n")
                        continue

                    print(f"---Loading: {image}")

                    if os.path.exists(markdown):
                        markdown_path = os.path.abspath(os.path.join(images, '..', f'{filename}.md'))
                    else:
                        markdown_path = docling_PDF2Text(pdf_path)

                    #print(markdown_path)
                    context = vlm_loadContext(markdown_path=markdown_path)

                    image_path = os.path.abspath(os.path.join(images, image))
                    print(f"Processing: {image_path}")

                    variant_count = -1
                    image_dict = None
                    for attempt in range(5):
                        response = vlm_ImageDescription(image_path=image_path, context=context)

                        try:
                            resp_dict = json.loads(response)
                        except:
                            continue

                        if not resp_dict:
                            continue

                        if resp_dict["variant_count"] is None:
                            resp_dict["variant_count"] = 0
                        elif resp_dict["variant_count"] == "null":
                            resp_dict["variant_count"] = 0

                        print(f"count = {resp_dict['variant_count']}")
                        count = int(resp_dict["variant_count"])

                        if count > variant_count:
                            image_dict = resp_dict
                            variant_count = count
                        else:
                            continue

                    image_dict['path'] = image_path

                    for key, value in image_dict.items():

                        if value is None:
                            if key in ["authors", "genes_mentioned", "variants"]:
                                image_dict[key] = []
                            elif key in ["title", "type", "figure_table_no", "sub_figure_table_no", "text",
                                         "caption", "description", "table", "variant_count"]:
                                image_dict[key] = ""
                            elif key == "variant_count":
                                image_dict[key] = 0

                        if key == "variants" and isinstance(value, list):
                            for variant in value:
                                for key, value in variant.items():
                                    if value is None:
                                        variant[key] = ""

                    print(json.dumps(image_dict, indent=4))

                    chunks = None

                    if image_dict['type'] == 'table':

                        chunks = splitter.split_text(
                            f"Caption: {image_dict['caption']} "
                            f"Description: {image_dict['description']} "
                            f"Table: {image_dict['table']}"
                        )

                    elif image_dict['type'] == 'figure':

                        chunks = splitter.split_text(
                            f"Caption: {image_dict['caption']} "
                            f"Description: {image_dict['description']}"
                        )

                    else:
                        chunks = splitter.split_text(
                            f"Caption: {image_dict['caption']} "
                            f"Description: {image_dict['description']}"
                        )

                    chunk_dicts = []

                    for i, chunk in enumerate(chunks):

                        weaviate_dict = {}

                        for key, value in image_dict.items():

                            weaviate_dict[key] = copy.deepcopy(value)

                        weaviate_dict["chunk_idx"] = i + 9001

                        weaviate_dict["chunk"] = chunk

                        chunk_genes = []

                        if weaviate_dict['type'] == 'table':

                            if len(weaviate_dict['genes_mentioned']) > 0:
                                for gene in weaviate_dict['genes_mentioned']:
                                    if gene in chunk:
                                        chunk_genes.append(gene)

                            weaviate_dict['genes_mentioned'] = chunk_genes

                            chunk_variants = []
                            if len(weaviate_dict['variants']) > 0:
                                for variant in weaviate_dict['variants']:
                                    nom_list = []
                                    c_variant = variant["transcript_variant"]
                                    g_variant = variant["genomic_variant"]
                                    p_variant = variant["protein_variant"]

                                    if c_variant and c_variant != "null":
                                        nom_list.append(c_variant)
                                    if g_variant and g_variant != "null":
                                        nom_list.append(g_variant)
                                    if p_variant and p_variant != "null":
                                        nom_list.append(p_variant)

                                    if len(nom_list) > 0:
                                        for nom in nom_list:
                                            if nom in chunk:
                                                chunk_variants.append(variant)
                                                break
                                    else:
                                        continue

                            weaviate_dict['variants'] = chunk_variants

                            weaviate_dict['variant_count'] = len(weaviate_dict['variants'])

                            weaviate_dict["path"] = image_path

                        print(json.dumps(weaviate_dict, indent=4))

                        with open("chunk_sizes.txt", "a") as f:
                            f.write(f"Image: {image}\n"
                                    f"Chunk No.: {i+9001}\n"
                                    f"Chunk length: {len(chunk)}\n"
                                    f"Dict length: {len(str(weaviate_dict))}\n")

                        chunk_dicts.append(weaviate_dict)

                    print(f"---Loading {image} into weaviate database.\n"
                          f"--- {chunk_dicts} chunks awaiting upload.")
                    weaviateImportImage(dict_list=chunk_dicts, model=model)
                    print(f"---{image} successfully loaded into weaviate database")

                    with open("image_files.txt", "a") as f:
                        f.write(f"{image}\n")

            if os.path.exists(images):
                shutil.rmtree(images)

            with open("pdf_paths.txt", "a") as f:
                f.write(f"{pdf_path}\n")



