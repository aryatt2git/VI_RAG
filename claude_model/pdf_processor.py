import pymupdf4llm
from typing import List, Dict
import re
import os
import time
import requests
from dataclasses import dataclass

### **Phase 1: Intelligent PDF Processing**

@dataclass
class VariantMention:
    variant_string: str
    normalized_forms: List[str]
    context_window: str
    page: int
    in_table: bool
    in_figure: bool
    pmid: str


class FHPDFProcessor:
    def __init__(self):
        self.variant_patterns = self._compile_variant_patterns()

    def _compile_variant_patterns(self):
        """Regex patterns for variant detection"""
        return {
            'hgvs_dna': r'c[.]([-*]*\d+|[-]*\d+_[-]*\d+|[-]*\d+[+-]\d+)([ACGTacgt]+>[ACGTacgt]+|delins[ACGTacgt]*(>[ACGTacgt]+)*|del[ACGTacgt]*|ins[ACGTacgt]*|dup[ACGTacgt]*|inv[ACGTacgt]*|\d+[ACGTacgt]{3}[[]\d+[]])',
            'hgvs_protein': r'p[.](\()*(0)*(\?)*[*]*[?]*(\d*[a-zA-Z]{3})*(\d+([a-zA-Z]{3})*(fs)*[*]*(\d+)*|\d*_[a-zA-Z]{3}\d+(ins)*[a-zA-Z]*|\d*_[a-zA-Z]{3}\d+(delins)*[a-zA-Z]*|\d+=|\d+[*]|ext\d*)*(\))*',
            'short_protein': r'[A-Za-z]\d+[A-Za-z]',
            'genomic': r'g[.]([-]*\d+|[-]*\d+_[-]*\d+|[-]*\d+[+-]\d+)([ACGTacgt]+>[ACGTacgt]+|delins[ACGTacgt]*(>[ACGTacgt]+)*|del[ACGTacgt]*|ins[ACGTacgt]*|dup[ACGTacgt]*|inv[ACGTacgt]*|\d+[ACGTacgt]{3}[[]\d+[]])',
            'dbsnp': r'rs\d+',
            # Add more patterns
        }

    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Multi-modal extraction with variant-aware chunking
        """
        chunks = []
        images_name = os.path.splitext(os.path.basename(pdf_path))[0]

        # 1. Extract structured content
        doc = pymupdf4llm.to_markdown(
            pdf_path,
            page_chunks=False,  # Don't chunk by page!
            write_images=True,
            image_path=f"./images/{images_name}/"
        )

        # 2. Separate processing for tables
        tables = self.extract_tables_enhanced(pdf_path)

        # 3. Separate processing for figures
        figures = self.extract_figures_with_captions(pdf_path)

        # 4. Process main text with variant-aware chunking
        text_chunks = self.variant_aware_chunking(doc, images_name)

        # 5. Process tables
        table_chunks = self.process_variant_tables(tables, images_name)

        # 6. Process figures with VLM
        figure_chunks = self.process_figures_vlm(figures, images_name)

        return pdf_path + text_chunks + table_chunks + figure_chunks



### **Phase 2: Variant-Aware Chunking**

class VariantAwareChunker:
    def __init__(self, chunk_size=512, overlap=128):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_with_variant_context(self, text: str, pdf_path: str) -> List[Dict]:
        """
        Ensure variants are never split and always have context
        """
        chunks = []

        # Detect all variants in text
        variants = self.variant_detector.find_all_variants(text)

        if not variants:
            # Standard semantic chunking if no variants
            return self.semantic_chunk(text)

        # Create chunks that preserve variant context
        for variant in variants:
            # Extract extended context around variant
            start_idx = max(0, variant.position - 300)
            end_idx = min(len(text), variant.position + 300)

            context = text[start_idx:end_idx]

            genes = ['LDLR', 'PCSK9', 'APOB', 'APOE', 'LDLRAP1']
            genes_mentioned = []

            for gene in genes:
                if gene in context:
                    genes_mentioned.append(gene)

            chunk = {
                'text': context,
                'variant_mentions': [variant.string],
                'pdf_path': pdf_path,
                'chunk_type': 'variant_context',
                # Critical: metadata for filtering
                'genes_mentioned': genes_mentioned,
                'has_functional_data': self.detect_functional_keywords(context),
                'has_phenotype_data': self.detect_phenotype_keywords(context),
            }

            chunks.append(chunk)

        return chunks


### **Phase 3: Enhanced Table Processing**

class TableVariantExtractor:
    """
    Tables often contain the highest density of variant information
    """

    def process_variant_table(self, table_df, pdf_path: str) -> List[Dict]:
        chunks = []

        # Detect variant columns
        variant_cols = self.identify_variant_columns(table_df)

        # Process each row as a separate chunk
        for idx, row in table_df.iterrows():
            variants_in_row = []

            # Extract variants from identified columns
            for col in variant_cols:
                variant_str = str(row[col])
                if self.is_variant(variant_str):
                    variants_in_row.append(variant_str)

            if not variants_in_row:
                continue

            # Create structured representation
            row_text = self.row_to_natural_language(row, table_df.columns)

            chunk = {
                'text': row_text,
                'table_data': row.to_dict(),
                'variant_mentions': variants_in_row,
                'pdf_path': pdf_path,
                'chunk_type': 'table_row',
                # Extract key columns
                'ldl_cholesterol': self.extract_ldl_value(row),
                'functional_effect': self.extract_functional_effect(row),
                'classification': self.extract_classification(row),
            }

            chunks.append(chunk)

        return chunks

    def identify_variant_columns(self, df) -> List[str]:
        """Intelligently detect which columns contain variants"""
        variant_cols = []

        for col in df.columns:
            col_lower = col.lower()

            # Check column headers
            if any(keyword in col_lower for keyword in
                   ['variant', 'mutation', 'hgvs', 'nucleotide', 'amino acid', 'cdna', 'protein']):
                variant_cols.append(col)
                continue

            # Check content
            sample_values = df[col].astype(str).head(5)
            variant_count = sum(
                1 for val in sample_values
                if self.is_variant(val)
            )

            if variant_count >= 2:
                variant_cols.append(col)

        return variant_cols

    def row_to_natural_language(self, row, columns) -> str:
        """Convert table row to searchable text"""
        parts = []
        for col, val in zip(columns, row):
            if pd.notna(val):
                parts.append(f"{col}: {val}")

        return ". ".join(parts)