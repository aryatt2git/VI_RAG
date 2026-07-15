"""
Functions in this script are used to eextract text, figures, and tables from .PDF files.
"""

from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TesseractCliOcrOptions,
)
import os
import re
from pathlib import Path
from logger import setup_logging
import logging

# Setup logging.
setup_logging()
logger = logging.getLogger(__name__)

def docling_PDF2Text(filepath:str):
    """
    This function is used to extract text from PDF files and store it in a markdown file. This includes text from the
    main body of PDF file, captions/legends and text held in tables.

    :params: filepath: The filepath to the PDF file.
    return os.path.abspath(markdown_filepath): The absolute filepath to the markdown file where the text is stored.
    """

    logger.info(f"Extracting text from PDF: {os.path.basename(filepath)}.")

    input_pdf = filepath
    base_name = os.path.splitext(os.path.basename(input_pdf))[0]
    pdf_name = base_name.strip().replace(" ", "_")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=True
    )
    ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
    pipeline_options.ocr_options = ocr_options

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    doc = converter.convert(input_pdf).document
    md = doc.export_to_markdown()
    markdown_filepath = f'{pdf_name}.md'

    delete_list = []
    for i, section in enumerate(md.split('##')):
        if section.lower().startswith(" references"):
            delete_list.append(i)
        if section.lower().startswith(" abstract"):
            delete_list.append(i)
        if section.lower().startswith(" acknowledgements") or section.lower().startswith(" acknowledgments"):
            delete_list.append(i)
        if section.lower().startswith(" conflicts of interest") or section.lower().startswith(" conflict of interest"):
            delete_list.append(i)

    with open(markdown_filepath, 'a') as f:
        for i, section in enumerate(md.split('##')):
            section.strip()
            if i not in delete_list:
                f.write(f"##{section}")

    logger.info("Successfully created Markdown file.")

    return os.path.abspath(markdown_filepath)


def docling_ImageExport(filepath:str):
    """
    This function is used to extract images from PDF files and store them in the images/ directory. This includes
    figures, and tables from the main body of the PDF file, as well as logos and miscellaneous images that be included
    in the PDF file.

    :params: filepath: The filepath to the PDF file.
    :return: os.path.abspath(images_dir): The absolute filepath to the images/ directory where the images are stored.
    """

    logger.info(f"Extracting Figures and Tables from PDF: {os.path.basename(filepath)}")

    input_pdf = filepath

    base_name = os.path.splitext(os.path.basename(input_pdf))[0]
    images_name = base_name.strip().replace(" ", "_")
    images_dir = f'./images/'
    if not os.path.exists(images_dir):
        os.makedirs(images_dir, exist_ok=True)

    # Keep page/element images so they can be exported. The `images_scale` controls
    # the rendered image resolution (scale=1 ~ 72 DPI). The `generate_*` toggles
    # decide which elements are enriched with images.
    IMAGE_RESOLUTION_SCALE = 2.0

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    conv_res = doc_converter.convert(input_pdf)
    doc_filename = conv_res.input.file.stem

    table_counter = 0
    picture_counter = 0
    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, TableItem):
            table_counter += 1
            element_image_filename = (
                    Path(images_dir) / f"{images_name}-table-{table_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")

        if isinstance(element, PictureItem):
            picture_counter += 1
            element_image_filename = (
                    Path(images_dir) / f"{images_name}-picture-{picture_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")

    # Save markdown with externally referenced pictures
    # md_filename = Path(images_dir) / f"{images_name}-with-image-refs.md"
    # conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)

    logger.info("Docling complete.")

    return os.path.abspath(images_dir)

"""
pdf_path = '/Users/arjun/PycharmProjects/VI_RAG/VI_RAG/tools/Hori et al 2019 PMID 31491741.pdf'
path = docling_PDF2Text(pdf_path)
docling_FigTableExport(pdf_path)
"""