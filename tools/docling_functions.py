from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TesseractCliOcrOptions,
)
import os
from pathlib import Path

def docling_PDF2Text(filepath:str):

    print("---Extracting text from PDF---")

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

    with open(markdown_filepath, 'w') as f:
        f.write(md)

    print("---markdown created successfully---")

    return os.path.abspath(markdown_filepath)


def docling_FigTableExport(filepath:str):

    print("---Extracting Figs and Tables---")

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

    print("---docling complete---")

    return os.path.abspath(images_dir)

"""
pdf_path = '/Users/arjun/PycharmProjects/VI_RAG/VI_RAG/tools/Hori et al 2019 PMID 31491741.pdf'
path = docling_PDF2Text(pdf_path)
docling_FigTableExport(pdf_path)
"""