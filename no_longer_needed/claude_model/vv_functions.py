import re
import os
import time
import json
import requests
from typing import List, Dict


def normalize_variant(variant_string: str) -> Dict:
    """
    Use VEP/Mutalyzer API to normalize variants
    Critical for handling Glu80Lys vs Glu101Lys issues
    """
    # Try multiple transcript references for FH genes
    transcripts = {
        'LDLR': 'NM_000527.5',
        'APOB': 'NM_000384.3',
        'PCSK9': 'NM_174936.4',
        'APOE': 'NM_000041.4',
        'LDLRAP1': 'NM_015627.3'
    }
    base_url_vv = "https://rest.variantvalidator.org/VariantValidator/variantvalidator/GRCh38/"
    normalized_forms = {}

    # Call VEP API or local VEP
    for gene, ref in transcripts.items():
        try:
            if ref == 'NM_174936.4':
                variant = f'{ref}:{variant_string}'
                refseq_variant = variant.replace(':', '%3A').replace('>', '%3E')
                url_vv = f"{base_url_vv}{refseq_variant}/mane?content-type=application%2Fjson"

                #result = self.vep_api.normalize(variant_string, ref)
                #normalized_forms[f'{gene}_{ref}'] = result

                for attempt in range(5):

                    try:
                        # Send an HTTP GET request to the API.
                        response = requests.get(url_vv)

                        # Raise an exception if the HTTP status code is not 200 (OK).
                        response.raise_for_status()

                        # The time module creates a 0.5s delay after each request to VariantValidator (VV), so that VV is not
                        # overloaded with requests.
                        time.sleep(0.5)

                        # Parse the API response into a Python dictionary.
                        data = response.json()

                        """
                        first_key = list(data.keys())[0]

                        c_variant = data[first_key]['hgvs_refseqgene_variant']
                        medium_p_variant = data[first_key]['hgvs_predicted_protein_consequence']['tlr']
                        short_p_variant = data[first_key]['hgvs_predicted_protein_consequence']['slr']
                        g_variant = data[first_key]['primary_assembly_loci']['grch38']['hgvs_genomic_description']
                        """

                        pp_json = json.dumps(data, indent=4)

                        print(pp_json)

                    except:
                        continue

            else:
                continue

        except:
            continue

    return normalized_forms

normalize_variant('c.301G>A')