import base64
import os
import json
from ollama import chat
from ollama import ChatResponse

def vlm_TextAnnotation(input_text, markdown_path):

    with open(markdown_path, "r", encoding='utf-8') as f:
        context = f.read()

    VLM_query = (
        f'Understand, analyse and interpret the following text using the the attached markdown file for context: '
        f'{input_text}'
        'Do not hallucinate. Do not speculate. Process the attached markdown file in accordance with the instructions. '
        'Return answer in accordance with the following JSON format:'
        '  {'
        '  "chunk": "", '
        '  "description": "", '
        '  "genes_mentioned": [], '
        '  "variant_count": "", '
        '  "variants": '
        '      {'
        '          "gene": "", '
        '          "genomic_variant": "", '
        '          "transcript_variant": "", '
        '          "exon": "", '
        '          "protein_variant": "", '
        '          "protein_domain": "", '
        '          "clinical_information": "", '
        '          "clinical_significance": "", '
        '          "frequency": "", '
        '          "hetero_carriers": "", '
        '          "homo_carriers": "", '
        '          "affected_carriers": "", '
        '          "unaffected_carriers": "", '
        '      }'
        '}'
        'Condense the information so that it can be easily imported into a Python dictionary using json.loads().'
        )

    augmented_prompt = f"""
    Use the following context to answer the question. 
    ---
    CONTEXT:
    {context}
    ---
    QUESTION: {VLM_query}

    INSTRUCTIONS: 
    - To string fields, return 'null' if information cannot be provided by content from the attached markdown file.
    - To a list field that cannot be populated, return [].
    - Return RAW JSON only. Do not include any introductory text, markdown code blocks (```), or concluding remarks.
    - Only use information from the attached markdown file to provide context to the text.
    - Never miss any text.
    - Analyse all of the input text.
    - Remove any whitespace from every variant nomenclature that appears in the text.
    - Never omit keys in the JSON. 
    - Never hallucinate.
    - Never speculate.
    - Do your best.
    - If you are unable to extract any clinically relevant information or cannot understand the information, let the user know.
    - To the 'chunk' key, assign the text as it appears in the input but remove any whitespace from every variant nomenclature that appears in the input text.
    - To the 'description' key, provide and assign a comprehensive analysis and description of the clinical information from the corresponding input text using the attached markdown file for context.
    - To the 'genes_mentioned' key, assign a list of the names of the genes mentioned in the corresponding input text.
    - To the 'variant_count' key, assign the number of variants that appear in the input text.
    - Do not count the genomic and proteomic variant descriptions of the same variant more than once.
    - To the 'variants' key, assign a nested dictionary for each and every variant mentioned in the corresponding input text.
    - Do not miss any variants out. 
    - To the 'gene' key, assign the gene symbol of the gene that the corresponding variant is in. Use the attached markdown file for context.
    - To the 'genomic_variant' key, assign the genomic variant as it appears in the corresponding text. It should start with 'g.' but might not. Remove any whitespace in the variant nomenclature.
    - To the 'transcript_variant' key, assign the variant described at the transcript level as it appears in the corresponding text. It should start with 'c.' but might not.  Remove any whitespace in the variant nomenclature.
    - To the 'exon' key, assign the exon of the gene that the corresponding variant is in. Use the attached markdown file for context.
    - To the 'protein_variant' key, assign the variant described at the protein level as it appears in the corresponding text. It should start with 'p.' but might not. Remove any whitespace in the variant nomenclature.
    - To the 'protein_domain' key, assign the protein domain of the protein that the corresponding variant is in. Use the attached markdown file for context.
    - To the 'clinical_information' key, provide and assign a comprehensive analysis and description of the clinical information related to the corresponding variant from the corresponding text. Use the attached markdown file for context.
    - To the 'clinical_significance' key, provide and assign a comprehensive analysis and description of the clinical significance of the corresponding variant from the the corresponding text. Use the attached markdown file for context.
    - To the 'frequency' key, assign the frequency of the corresponding variant as described in corresponding text.
    - To the 'hetero_carriers' key, assign the number of affected carriers with the corresponding variant in a heterozygous genotype counted in the corresponding text.
    - To the 'homo_carriers' key, assign the number of affected carriers with the corresponding variant in a homozygous genotype counted in the corresponding text.
    - To the 'affected_carriers' key, assign the total number of affected carriers with the corresponding variant counted in the corresponding text.
    - To the 'unaffected_carriers' key, assign the total number of unaffected carriers with the corresponding variant counted in the corresponding text.
    """

    response: ChatResponse = chat(
        model='qwen3.5:397b-cloud',
        messages=[
            {
                'role': 'system',
                'content': 'You are the best genomic clinical scientist ever because of your ability '
                           'to understand, analyse and interpret the content of research papers/articles/literature '
                           'stored in markdown files. Use only the input text and the attached markdown file for '
                           'context. Do not hallucinate. Do not speculate.'
            },
            {
                'role': 'user',
                'content': augmented_prompt
            }
        ],
        options={
            'temperature': 0.0,  # Lower temperature makes the output more focused and consistent
            'repeat_penalty': 1.0,  # Setting to 1.0 turns OFF the penalty that stops it from repeating itself
            'num_predict': 25000,  # Give the model plenty of room to write a long answer
            'seed': 42,  # Uncomment this if you want the EXACT same answer every single time
            'num_ctx': 32768,  # Add this to handle the image + markdown
        },
        keep_alive=0  # This forces a fresh start for the next run
    )

    print("--- VLM RESPONSE ---")
    # print(response['message']['content'])
    # or access fields directly from the response object
    #print(response.message.content)
    print(f"Query costed: {response.prompt_eval_count} tokens")
    print(f"Response costed: {response.eval_count} tokens")
    print(f"Total cost: {response.prompt_eval_count + response.eval_count} tokens")

    return response.message.content

texts=[
    "In 801 Japanese FH patients, 137 LDLR variants were identified, and 92, 44 and 1 of these variants were classified as pathogenic, VUS, and benign, respectively. Pathogenic variants in the LDLR and PCSK9 genes were found in 46% (n = 296) and 7.8% (n = 51) of unrelated FH patients (n = 650), respectively. Sixty of the 92 LDLR pathogenic variants were identified in ClinVar, and 32 were newly identified in this study. The proportion of LDLR pathogenic variants was high in patients with a younger age of CAD onset and significantly decreased with increases in the age of onset of CAD. The proportion of male patients harbouring LDLR pathogenic variants significantly decreased with increases in the age of CAD onset, but that of female patients did not change with increases in the age of onset of CAD. One possible explanation might be that a lower number of female patients developed CAD compared with the number of male patients. The LDLR pathogenic variants were clustered in exons 4 and 10. The variants in the LDLR gene were previously shown to be clustered in exon 4, which is the largest exon in the LDLR gene. In the present study, 17% (n = 110) of unrelated FH patients contained at least one of five frequent LDLR variants, including c.1845+2T > C,c.1012T > A: p.(Cys338Ser), c.1297G > C: p.(Asp433His), c.1702C > G: p. (Leu568Val) and c.2431A > T: p.(Lys811*). In the Hokuriku district, the following three frequent LDLR variants were found in 44.5% of the FH patients (n = 1054): c.2431A > T: p.(Lys811*) (n = 292; 28%), c.2312-3C > A (n = 119; 11%), and deletion of exon 2-3 (n = 58; 5.5%). Yu and Mabuchi et al. reported that the c.2431A > T: p.(Lys811*) variant is an example of a 'founder effect' or a variant that prevailed both locally and widely over a long period of time. In the present study, 67.5% of FH patients lived in the Kansai district, which is one of the largest metropolitan areas in Japan and has a greater influx of people, and the rest lived throughout Japan. Therefore, this study could reflect the distribution of LDLR variants across Japan. The clinical phenotypes of FH in carriers of each of the five frequent LDLR pathogenic variants were compared with those of the noncarriers in patients harbouring LDLR pathogenic variants. Carriers of the c.1845+2T > C and c.1702C > G: p.(Leu568Val) variants had significantly lower serum LDL-C levels and higher serum HDL-C levels than the noncarriers. The serum TC levels in carriers of the c.1297G > C: p.(Asp433His) variant were significantly higher than those in the noncarriers. The prevalence of ATT was significantly higher in carriers of the c.1012T > A: p.(Cys338Ser) variant and significantly lower in carriers of the c.1702C > G: p.(Leu568Val) variant than in the noncarriers. We recently reported that an increased ATT is associated with decreased HDL-C levels, a decreased cholesterol efflux capacity or an increased prevalence of hypertension in FH patients. Thus, after adjusting for the covariates, the prevalence of ATT was not associated with the c.1702C > G: p.(Leu568Val) variant or the c.1012T > A: p.(Cys338Ser) variant. The c.1702C > G: p",
    ". The prevalence of ATT was significantly higher in carriers of the c.1012T > A: p.(Cys338Ser) variant and significantly lower in carriers of the c.1702C > G: p.(Leu568Val) variant than in the noncarriers. We recently reported that an increased ATT is associated with decreased HDL-C levels, a decreased cholesterol efflux capacity or an increased prevalence of hypertension in FH patients. Thus, after adjusting for the covariates, the prevalence of ATT was not associated with the c.1702C > G: p.(Leu568Val) variant or the c.1012T > A: p.(Cys338Ser) variant. The c.1702C > G: p. (Leu568Val) variant has previously shown to yield a mild phenotype based on a high level of LDLR activity, which is consistent with the results of the present study. The c.1845+2T > C variant has been reported to be a receptor-negative variant. The c.1297G > C: p. (Asp433His) variant is associated with impaired processing and rapid degradation of the synthesized receptor. In the future, it is necessary to assess the effect of each variant on phenotype using a large number of samples. We defined c.94G > A: p.(Glu32Lys), c.385G > A: p.(Asp129Asn), c.644G > A: p.(Arg215His), and c.1486C > T: p.(Arg496Trp) as PCSK9 pathogenic variants. The prevalence of ATT, the main characteristic of FH, in the patients harbouring PCSK9 pathogenic variants was 44%, and this value was significantly lower than that observed for the patients harbouring LDLR pathogenic variants. In the present study, patients harbouring the c.94G > A: p.(Glu32Lys) variant comprised 88% (n = 45) of unrelated patients harbouring PCSK9 pathogenic variants (n = 51). The LDL-C levels and the prevalence of ATT in patients harbouring LDLR pathogenic variants were significantly higher than those in patients harbouring the c.94G > A: p.(Glu32Lys) variant in the PCSK9 gene. The c.94G > A: p.(Glu32Lys) variant in the PCSK9 gene is frequently detected in Japanese FH patients and is a specific variant in East Asian population. Mabuchi et al. has also reported that FH patients harbouring the c.94G > A: p.(Glu32Lys) variant have mild phenotypes compared with FH patients harbouring LDLR mutations. In an in vitro study, Noguchi et al. have reported that HepG2 cells transfected with PCSK9-p.(Glu32Lys) secreted significantly larger amounts of PCSK9 into the media than those with PCSK9-WT after 24 h of incubation (139% + 13% vs. 100% + 3%, p < 0.01). Thus, the phenotype of FH patients harbouring the c.94G > A: p.(Glu32Lys) variant differs from that of patients harbouring a gain-of-function mutation, including the c.1120G > T: p.(Asp374Tyr) variant in the PCSK9 gene detected in the Western countries. The c.94G > A: p.(Glu32Lys) variant in the PCSK9 gene is defined as pathogenic but shows a mild FH phenotype compared with gain-of-function mutations in the PCSK9 gene detected in other countries. It was recently reported that the c.94G > A: p.(Glu32Lys) variant in the PCSK9 gene affects the age of onset of myocardial infarction by increasing the LDL-C levels in the Japanese general population",
    ". The c.94G > A: p.(Glu32Lys) variant in the PCSK9 gene is defined as pathogenic but shows a mild FH phenotype compared with gain-of-function mutations in the PCSK9 gene detected in other countries. It was recently reported that the c.94G > A: p.(Glu32Lys) variant in the PCSK9 gene affects the age of onset of myocardial infarction by increasing the LDL-C levels in the Japanese general population. It is thus likely that FH might be underdiagnosed in patients with the c.94G > A: p.(Glu32Lys) variant in the PCSK9 gene, despite their genetic risk for CAD. We previously reported that the c.10G > A: p.(Val4lle) variant in the PCSK9 gene increases the prevalence of CAD in accordance with an elevation of the LDL-C level in carriers of an LDLR variant. However, in the present study, the c.10G > A: p.(Val4lle) variant in the PCSK9 gene was classified as benign according to the ACMG/AMP guidelines, and the allele frequency in unrelated HeFH patients was lower than that in the Japanese general population. In the Korean population, the p.(Val4Ile) variant in the PCSK9 gene was detected in an individual whose untreated LDL-C levels were < 48 mg/dL. The c.10G > A: p.(Val4lle) variant in the PCSK9 gene has been proposed to not affect the phenotype of HeFH by itself. We predicted the risk of CAD from a genetic analysis of HeFH. In the present study, LDLR pathogenic variants were found in 66.7% (n = 24) of the patients who developed CAD at less than 40 years of age and were still observed in 37.7% of patients who developed CAD at 60 years of age or older. Some standard risk factors for CAD, including diabetes, hypertension, and smoking, have been shown to be associated with an increased risk for CAD in HeFH patients. In the present study, these risk factors in HeFH patients are expected to increase the risk for developing CAD at a relatively advanced age. Thus, it has been hypothesized that LDLR pathogenic variants are related to the development of CAD at a younger age, and the other risk factors for CAD are related to the development of CAD at an advanced age. The genetic analysis of FH is very important for the identification of FH patients at high risk for premature CAD and for improving the prognosis of FH. This study has some limitations. First, this was a retrospective observational study, and thus, the subjects with missing clinical information were excluded from the analysis of the association of clinical phenotype with pathogenic variants. The exclusion of these patients might have caused a selection bias for the patient phenotype in contrast to genetic data. However, this study was based on one of the largest cohorts of Japanese FH patients and showed that the distribution of LDLR pathogenic variants differed from that found in the Hokuriku district, which has a large cohort of Japanese FH patients. Therefore, our results might reflect the distribution of LDLR pathogenic variants throughout Japan compared with the Hokuriku district. A second limitation is that only CAD events were considered; this cohort included both patients under treatment and those not receiving treatment, and the treatment period was also not considered. A detailed analysis of CAD events with proper consideration of the treatments and their durations is currently being conducted. A third limitation is that we did not analyse introns in the LDLR gene and the other causative genes. Unexplained FH might be partly explained by the accumulation of common SNPs and other causative genes. The fourth limitation is that our center specializes in CAD; thus, the prevalence of CAD in FH patients at our center might be higher than that in the general Japanese population of FH patients",
    ". Therefore, our results might reflect the distribution of LDLR pathogenic variants throughout Japan compared with the Hokuriku district. A second limitation is that only CAD events were considered; this cohort included both patients under treatment and those not receiving treatment, and the treatment period was also not considered. A detailed analysis of CAD events with proper consideration of the treatments and their durations is currently being conducted. A third limitation is that we did not analyse introns in the LDLR gene and the other causative genes. Unexplained FH might be partly explained by the accumulation of common SNPs and other causative genes. The fourth limitation is that our center specializes in CAD; thus, the prevalence of CAD in FH patients at our center might be higher than that in the general Japanese population of FH patients. In conclusion, this study annotated the clinical significance of variants in the LDLR and PCSK9 genes and assessed the impact of their pathogenic variants on the clinical phenotype of Japanese HeFH patients. This study could provide important data for the highly accurate genetic diagnosis of Japanese FH patients."
]




filepath = '/Users/arjun/PycharmProjects/VI_RAG/VI_RAG/tools/'

for file in os.listdir(filepath):
    if file.endswith('.md'):
        markdown_path = os.path.abspath(file)
        print(markdown_path)

        for text in texts:
            response = vlm_TextAnnotation(input_text=text, markdown_path=markdown_path)
