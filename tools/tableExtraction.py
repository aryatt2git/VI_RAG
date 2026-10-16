import json
from vlm_functions import *
from docling_functions import *

# Temporary filepath to References directory where the .pdf files should be kept.
References = '/Users/arjun/PycharmProjects/VI_RAG/VI_RAG/tools/References/'

# Iterate through each file in every directory and subdirectory in the 'References/' folder.
for root, dir, files in os.walk(References, topdown=False):
    for file in files:
        # Specify only .pdf files.
        #if file.endswith('.pdf'):
        if file.endswith('DiCostanzo2021 PMID 34756585 supp info 1.docx'):

            # Get the path to the file currently being iterated over.
            pdf_path = os.path.join(os.path.abspath(root), file)

            # Retrieve the .pdf filename and use it as the name for the markdown file.
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            markdown_name = f'{base_name.strip().replace(" ", "_")}.md'

            # The docling_PFD2Text generates a markdown file from the document's filepath. The filepath to the markdown
            # file is returned by the docling_PDF2Text function and assigned to the 'markdown_path' variable.
            markdown_path = docling_PDF2Text(pdf_path)

            # Create an empty variable to store the list of python dictionaries in JSON format produced by the VLM.
            ext_json_list = None

            # Make 10 attempts to retrieve a response from the VLM until a response has been retrieved.
            for attempt in range(10):

                # If ext_json_list has been populated, break the loop.
                if ext_json_list:
                    break

                # Sometimes the VLM does not produce a response that can be loaded into Python without raising
                # an exception. Therefore, a simple try and except statement is used to retry parsing a list of
                # each section from the .pdf into ext_json_list.
                try:
                    # The vlm_TextExtraction function is used to parse each section in the .pdf file into a JSON
                    # object along with accompanying meta-data. Each section if then store in a list.
                    ext_response = vlm_TableExtractor(markdown_path=markdown_path)
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