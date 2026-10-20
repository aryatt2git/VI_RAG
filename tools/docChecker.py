import os
import re
import json
import shutil

import pandas as pd
from vlm_functions import vlm_docChecker
from docling_functions import docling_PDF2Text

df = pd.read_csv("/data/home/arjun/git_repositories/pubmed_cli/output/FH_Ref_PMIDs_classified.csv")

pmid_df = df[df["detected_license_type"].isin([
    "CC BY",
    "Public domain",
    "Open access, non-commercial terms (unspecified license)",
    "Text/data mining or fair-use access only",
    #"CC BY-ND"
    ]
)]

cpr_pmids = pmid_df["PMID"].unique().tolist()
str_pmids_ls = []

for pmid in cpr_pmids:
    str_pmids_ls.append(str(pmid))

"""
for root, dir, files in os.walk(references, topdown=False):
    for file in files:

        re_match = re.search(r'PMID\s(\d+)', file)

        if (re_match is not None):
            pmid = re_match.group(1).strip()

            if pmid not in pmids:
                pmids.append(pmid)
"""

references = "/mnt/F_share/DNA/Cardiac/Familial Hypercholesterolaemia/References/"
destination = "/data/home/arjun/FH_references/"
full_pdf_dict = {}
other_count = 0

for pmid in str_pmids_ls:

    pmid_filepaths = []

    for root, dir, files in os.walk(references, topdown=False):
        for file in files:

            if pmid in file and "sun" in file:
                markdown_path = docling_PDF2Text(os.path.join(root, file))
                pmid_filepaths.append(markdown_path)

    print(f"PMID {pmid}:")
    print(f"PMID Files:\n{pmid_filepaths}")

    response = vlm_docChecker(pmid_filepaths, pmid)
    clean_response = response.replace("```json", "").replace("```", "")
    json_resp = json.loads(clean_response)

    print(json.dumps(json_resp, indent=4))

    for key, value in json_resp.items():
        if value != "copy":
            shutil.copy(key, destination)

        if value == "other":
            other_count += 1
            print("Other:", key)

    full_pdf_dict[pmid] = json_resp

    for markdown in pmid_filepaths:
        os.remove(markdown)

file = open("pmid_json.txt", "w")
file.write(json.dumps(full_pdf_dict))
file.close()

print("Other counter:", other_count)