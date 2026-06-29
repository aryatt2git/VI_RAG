import hashlib

def generateTextID(chunk_dict):

    title = chunk_dict['title']
    authors = ", ".join(chunk_dict['authors'])
    path = chunk_dict['path']
    section_header = chunk_dict["section_header"]
    subsection_header = chunk_dict["subsection_header"]
    sub_subsection_header = chunk_dict["sub_subsection_header"]
    chunk_idx = chunk_dict["chunk_idx"]

    unique_str = f"{title}-{authors}-{path}-{section_header}-{subsection_header}-{sub_subsection_header}-{chunk_idx}"

    return hashlib.md5(unique_str.encode("utf-8")).hexdigest()


def generateImageID(chunk_dict):

    title = chunk_dict['title']
    authors = ", ".join(chunk_dict['authors'])
    path = chunk_dict['path']
    type = chunk_dict["type"]
    fig_table_no = chunk_dict["figure_table_no"]
    sub_figure_table_no = chunk_dict["sub_figure_table_no"]
    chunk_idx = chunk_dict["chunk_idx"]

    unique_str = f"{title}-{authors}-{path}-{type}-{fig_table_no}-{sub_figure_table_no}-{chunk_idx}"

    return hashlib.md5(unique_str.encode("utf-8")).hexdigest()

