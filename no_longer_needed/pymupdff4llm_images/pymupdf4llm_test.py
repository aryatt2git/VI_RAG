import pymupdf4llm
from typing import List, Dict
import os

def process_pdf(pdf_path: str) -> List[Dict]:

    chunks = []
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    images_name = base_name.strip().replace(" ", "_")

    images_dir = f'./images/{images_name}'

    if not os.path.exists(images_dir):
        os.makedirs(images_dir, exist_ok=True)

    doc = pymupdf4llm.to_markdown(
                pdf_path,
                page_chunks=False,  # Don't chunk by page!
                write_images=True,
                image_path=images_dir
            )

    with open("test.md", "w") as f:
        f.write(doc)

process_pdf('/Users/arjun/PycharmProjects/VI_RAG/VI_RAG/tools/Hori et al 2019 PMID 31491741.pdf')