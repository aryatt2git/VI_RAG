import os
import re
import json
from vlm_functions import vlm_docChecker
from docling_functions import docling_PDF2Text

pmids = []

for root, dir, files in os.walk("References", topdown=False):
    for file in files:

        re_match = re.search(r'PMID\s(\d+)', file)

        if (re_match is not None):
            pmid = re_match.group(1).strip()

            if pmid not in pmids:
                pmids.append(pmid)

for pmid in pmids:

    pmid_filepaths = []

    for root, dir, files in os.walk("References", topdown=False):
        for file in files:

            if pmid in file:
                markdown_path = docling_PDF2Text(os.path.join(root, file))
                pmid_filepaths.append(markdown_path)

    print(f"PMID {pmid}:")
    print(f"PMID Files:\n{pmid_filepaths}")

    response = vlm_docChecker(pmid_filepaths, pmid)
    clean_response = response.replace("```json", "").replace("```", "")
    json_resp = json.loads(clean_response)

    print(json.dumps(json_resp, indent=4))

    for markdown in pmid_filepaths:
        os.remove(markdown)