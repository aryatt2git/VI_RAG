# This example uses the GQL GraphQL client library.
#
# To install: pip3 install gql
#
# GQL is one popular Python GraphQL client, but there are others.
# See https://graphql.org/community/tools-and-libraries/?tags=python_client
import json
import time
import asyncio
import requests
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport


async def gnomad_query():

    gene_symbol = 'LDLR'
    variant = 'c.301G>A'

    vv_url = 'https://rest.variantvalidator.org/VariantValidator/'

    gene_url_vv = f"{vv_url}tools/gene2transcripts/{gene_symbol}?content-type=application%2Fjson"
    gene_response = requests.get(gene_url_vv)

    # Raise an exception if the HTTP status code is not 200 (OK).
    gene_response.raise_for_status()

    # The time module creates a 0.5s delay after each request to VariantValidator (VV), so that VV is not
    # overloaded with requests.
    time.sleep(0.5)

    # Parse the API response into a Python dictionary.
    gene_data = gene_response.json()

    for transcript_record in gene_data["transcripts"]:
        if transcript_record["annotations"]["mane_select"]:
            # Extract the NM_ number of the MANE select transcript.
            transcript_ref = transcript_record["reference"]

    hgvs_nom = f'{transcript_ref}:{variant}'
    variant_a, variant_b = variant.split(">")

    vv_query = f'{vv_url}variantvalidator/GRCh38/{transcript_ref}%3A{variant_a}%3E{variant_b}/mane?content-type=application%2Fjson'

    variant_response = requests.get(vv_query)

    # Raise an exception if the HTTP status code is not 200 (OK).
    variant_response.raise_for_status()

    # The time module creates a 0.5s delay after each request to VariantValidator (VV), so that VV is not
    # overloaded with requests.
    time.sleep(0.5)

    # Parse the API response into a Python dictionary.
    variant_data = variant_response.json()

    chr = variant_data[hgvs_nom]["primary_assembly_loci"]["grch38"]["vcf"]["chr"]
    pos = variant_data[hgvs_nom]["primary_assembly_loci"]["grch38"]["vcf"]["pos"]
    ref = variant_data[hgvs_nom]["primary_assembly_loci"]["grch38"]["vcf"]["ref"]
    alt = variant_data[hgvs_nom]["primary_assembly_loci"]["grch38"]["vcf"]["alt"]

    gnomad_key = f'{chr}-{pos}-{ref}-{alt}'

    transport = AIOHTTPTransport(url="https://gnomad.broadinstitute.org/api")
    client = Client(transport=transport, fetch_schema_from_transport=True)

    # For brevity, and to keep the focus on the Python code, we don't include every
    # field from the raw query here.

    gene_query = gql(
        """
        query VariantsInGene {
          gene(gene_symbol: "LDLR", reference_genome: GRCh38) {
            variants(dataset: gnomad_r4) {
              variant_id
              pos
              exome {
                ac
                ac_hemi
                ac_hom
                an
                af
              }
            }
          }
        }
    """
    )

    variant_query = gql(
        """
        query GetVariant($id: String!) {
            variant(variantId: $id, dataset: gnomad_r4) {
                exome {
                    af
                    ac
                    an
                    ac_hemi
                    ac_hom
                    populations {
                        id
                        ac
                        an
                        ac_hemi
                        ac_hom
                    }
                }
                genome {
                    af
                    ac
                    an
                    ac_hemi
                    ac_hom
                    populations {
                        id
                        ac
                        an
                        ac_hemi
                        ac_hom
                    }
                }
                joint {
                    ac
                    an
                    homozygote_count
                    hemizygote_count
                    populations {
                        id
                        ac
                        an
                        homozygote_count
                        hemizygote_count
                    }
                    filters
                }
            }
        }     
        """
    )

    # Execute the query on the transport
    params = {"id": gnomad_key}
    results = await client.execute_async(variant_query, variable_values=params)
    # print(results)

    # for result in results: ["gene"]["variants"]:
        # if result["variant_id"] == gnomad_key:
            # print(result)
            # return result

    for dict in results["variant"].items():
        dict_pp = json.dumps(dict, indent=4)
        print(dict_pp)


if __name__ == "__main__":
    asyncio.run(gnomad_query())