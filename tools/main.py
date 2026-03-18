import os
import json
import time
from chunk_text import chunkText
from vlm_functions import vlm_TextExtraction, vlm_TextAnnotation, vlm_Images
from weaviate_functions import loadWeaviate, weaviateImportText

filepath = '/Users/arjun/PycharmProjects/VI_RAG/VI_RAG/tools/'

model = loadWeaviate()
splitter = chunkText()

for file in os.listdir(filepath):
    time_1 = time.time()
    if file.endswith('.md'):
        markdown_path = os.path.abspath(os.path.join(filepath, file))
        print(markdown_path)

        chunk_dicts = []

        ext_response = vlm_TextExtraction(markdown_path=markdown_path)
        for dict in json.loads(ext_response):

            chunks = splitter.split_text(dict['text'])

            for i, chunk in enumerate(chunks):

                pdf_dict = {}

                pdf_dict["path"] = markdown_path

                for key, value in dict.items():
                    if key != 'text':
                        pdf_dict[key] = value

                pdf_dict["chunk_idx"] = i + 1

                anno_response = vlm_TextAnnotation(input_text=chunk, markdown_path=markdown_path)
                anno_dict = json.loads(anno_response)

                if anno_dict["variant_count"] == "null":
                    anno_dict["variant_count"] = 0

                for key, value in anno_dict.items():
                    pdf_dict[key] = value

                print(json.dumps(pdf_dict, indent=4))

                chunk_dicts.append(pdf_dict)

        weaviateImportText(dict_list=chunk_dicts, model=model)

        time_2 = time.time()
        print(f'{time_2 - time_1:.2f}')


for image in os.listdir(images):
    print(image)
    if image.endswith('.png'):
        filename = image.split('-')[0]
        markdown_path = os.path.abspath(os.path.join(images, '..', '..', f'{filename}.md'))
        print(markdown_path)
        image_path = os.path.abspath(os.path.join(images, image))
        print(f"Processing: {image_path}")

        variant_count = 0
        image_dict = None
        for attempt in range(5):
            response = vlm_Images(image_path=image_path, markdown_path=markdown_path)

            try:
                resp_dict = json.loads(response)
            except:
                continue

            if resp_dict["variant_count"] == "null":
                resp_dict["variant_count"] = 0

            count = int(resp_dict["variant_count"])

            if count > variant_count:
                image_dict = resp_dict
            else:
                continue

        image_dict['path'] = image_path
        print(json.dumps(image_dict, indent=4))

