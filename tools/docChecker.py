import os
import re
import json
import shutil
import pandas as pd
from subprocess import run
from doc2docx import convert
from vlm_functions import vlm_docChecker
from docling_functions import docling_PDF2Text

ref_pmid_ls = []

for root, dir, files in os.walk("References"):
    for file in files:
        re_match = re.search(r'PMID\s(\d+)', file)

        if (re_match is not None):
            pmid = re_match.group(1).strip()
            ref_pmid_ls.append(pmid)

csv_path = "/Users/arjun/PycharmProjects/VI_RAG/VI_RAG/tools/FH_Ref_PMIDs_classified.csv"
references = "/Users/arjun/Downloads/References 3/"
destination = "/Users/arjun/PycharmProjects/VI_RAG/VI_RAG/tools/References/"

#references = "/mnt/F_share/DNA/Cardiac/Familial Hypercholesterolaemia/References/"
#destination = "/data/home/arjun/FH_references/"
#csv_path = "/data/home/arjun/git_repositories/pubmed_cli/output/FH_Ref_PMIDs_classified.csv"

df = pd.read_csv(csv_path)

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

full_pmid_dict = {}
other_count = 0
unused_files = []

for pmid in str_pmids_ls:

    if pmid in ref_pmid_ls:
        continue

    # 31589614 files are too big for the prompt.
    # 35460704 The prompt is too long: 343387, model maximum context length: 262144.
    # 32041611 Prompt is too large to process.
    # 39020067 File format not allowed: Elfatih et al 2024 PMID 39020067 Supp Table 1.xls (you need xlsx).
    # 32820175 The prompt is too long: 428262, model maximum context length: 262144
    # 30583242 Prompt is too large to process.
    # 9143924 Input document Jensen et al 1997 PMID 9143924.pdf is not valid.
    # 35177841 ollama._types.ResponseError: http: request body too large (status code: 400)
    # 269629 The prompt is too long: 269629, model maximum context length: 262144
    if pmid in ["31589614", "35460704", "32041611", "39020067", "32820175", "30583242", "9143924", "35177841"]:
        continue

    pmid_filepaths = []
    markdown_filepaths = []

    for root, dir, files in os.walk(references, topdown=False):
        for file in files:

            if file.endswith(".zip") or "~$2015 PMID 25487149 supp info" in file:
                unused_files.append(file)
                continue

            filepath = os.path.join(root, file)

            if pmid in file:

                if file.endswith(".doc") or file.endswith(".DOC"):
                    basename = os.path.splitext(file)[0]
                    docx_path = os.path.join(root, f"{basename}.docx")

                    # If Microsoft Word is installed, doc2docx Python library can be used.
                    #convert(filepath, docx_path)
                    # If LibreOffice is installed, use Subprocess Python library to execute soffice.
                    soffice = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
                    run([soffice, "--headless", "--convert-to", "docx", "-outdir", root, filepath], check=True)

                    markdown_path = docling_PDF2Text(docx_path)
                    pmid_filepaths.append(docx_path)
                    markdown_filepaths.append(markdown_path)

                else:
                    markdown_path = docling_PDF2Text(filepath)
                    pmid_filepaths.append(filepath)
                    markdown_filepaths.append(markdown_path)

    if len(pmid_filepaths) == 0:
        continue

    print(f"PMID {pmid}:")
    print(f"PMID Files:\n{pmid_filepaths}")

    response = vlm_docChecker(markdown_filepaths, pmid)
    clean_response = response.replace("```json", "").replace("```", "")
    json_resp = json.loads(clean_response)

    print(json.dumps(json_resp, indent=4))

    filepath_dict = {}

    for key, value in json_resp.items():

        for i in range(len(markdown_filepaths)):
            if key == markdown_filepaths[i]:
                print("Markdown position path:", markdown_filepaths[i])
                print("File position path:", pmid_filepaths[i])

                filepath_dict[pmid_filepaths[i]] = value

            if value != "copy":
                shutil.copy(pmid_filepaths[i], destination)

            if value == "other":
                other_count += 1
                print("Other:", key)

    full_pmid_dict[pmid] = filepath_dict

    for file in os.listdir("/Users/arjun/PycharmProjects/VI_RAG/VI_RAG/tools/"):
        if file.endswith(".md"):
            os.remove(file)

file = open("pmid_json.txt", "w")
file.write(json.dumps(full_pmid_dict))
file.close()

print("Other counter:", other_count)
print("Unused files:\n")
for file in unused_files:
    print(file)