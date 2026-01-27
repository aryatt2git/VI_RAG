from typing import List, Dict
from transformers import AutoTokenizer

def chunk_text_tokens(
    title,
    section,
    text,
    tokenizer,
    chunk_size=250,
    overlap=25,
    add_special_tokens=False
) -> List[Dict]:
    """
    Chunk text into overlapping token-based chunks.

    Args:
        text (str): Input text.
        tokenizer: Hugging Face tokenizer.
        chunk_size (int): Tokens per chunk.
        overlap (int): Overlapping tokens between chunks.
        add_special_tokens (bool): Whether to add special tokens ([CLS], [SEP]).

    Returns:
        list[str]: List of decoded text chunks.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Tokenize once
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_tensors=None
    )["input_ids"]

    chunks = []
    chunk_id = 1
    step = chunk_size - overlap

    for i in range(0, len(token_ids), step):

        chunk_ids = token_ids[i:i + chunk_size]

        if len(chunk_ids) < overlap:
            break

        # Optionally add special tokens per chunk
        if add_special_tokens:
            chunk_tokens = tokenizer.build_inputs_with_special_tokens(chunk_ids)

        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        chunks.append({
            'chunk_id': chunk_id,
            'input_ids': chunk_ids,
            'attention_mask': [1] * len(chunk_ids),
            'title': title,
            'section': section,
            'genes': ['LDLR'],
            'text': chunk_text
        })

        chunk_id += 1

    return chunks

'''
tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

results = chunk_text_tokens(
    'Update of Japanese common LDLR gene mutations and their phenotypes: Mild type mutation L547V might predominate in the Japanese population',
    'Results',
    'Through the examination of the coding and promoter sequences, a small mutation of less than 25 bp in the LDLR gene was identified in 118 patients, representing 57.6% of the 205 unrelated FH heterozygotes. Fifty-three different mutations were identified in total. The nomenclature for the description of sequence variations followed established recommendations. For cDNA sequences, the A of the initiating ATG codon is denoted as nucleotide +1, and amino acids are numbered according to established conventions, with the N-terminal alanine of the mature form numbered as +1. Among the 53 total mutations, 23 had already been reported in Japanese subjects, while the remaining 30 were new Japanese mutations. There was variety in the types of mutations, including missense mutations, nonsense mutations, splicing mutations, deletions, and insertions, with missense mutations comprising 60% of the total. The mutation sites were distributed widely over the entire LDLR gene. There were eight relatively frequent mutations with percentages above 2.4%: C317S, c.1845+2T>C, K790X, L547V, P664L, D412H, c.2313-3C>A, and V776M. These mutations were classified as common mutations. The total number of patients carrying these common mutations was 65, comprising 31.7% of the 205 FH heterozygotes. The remaining 45 mutations were rare, being encountered in only one to three cases, and were present in 25.9% of the patients. For cases in which no mutation was found by these methods, MLPA analysis was performed to detect large DNA rearrangements. At least 10 kinds of deletions were identified, and no duplication or insertion was detected. A total of 21 patients had large deletions, comprising 10.2% of the subjects investigated, a percentage similar to that reported in other studies. Several deletions had been previously reported, while six kinds of deletions had not yet been reported in Japanese subjects. Among the 205 FH heterozygotes, deletions of exon 1 and of exons 2–3 were relatively frequent. At present, the precise deletion sites have not been clarified, so it is not known whether these deletion mutations are uniform, but deletions of exon 1 and exons 2–3 may represent common mutations in Japanese FH heterozygotes. In the remaining 32.2% of patients, no LDLR gene mutation was detected by the present analyses. The clinical features and plasma lipid levels of patients carrying each common LDLR mutation showed no gross differences except for those with the L547V mutation. Heterozygotes with L547V had lower LDL and total cholesterol levels compared with heterozygotes carrying other mutations, while triglyceride and HDL cholesterol levels did not differ significantly except for a lower HDL cholesterol level observed in heterozygotes with C317S. The frequency of xanthomas and the frequency of coronary artery disease in patients over 35 years old were also lower among heterozygotes with L547V, and the frequency of coronary artery disease was significantly lower than in heterozygotes with certain other mutations. These observations indicate that L547V is a mild phenotype mutation. Individual heterozygotes with L547V showed some variability in lipid levels, but average values were lower than those observed in heterozygotes with other mutations, although still higher than in normocholesterolemic individuals, suggesting an elevated coronary artery disease risk compared with the general population. One heterozygote had already developed coronary artery disease by the age of 53. The clinical phenotype of L547V was further compared with other common mutations in true homozygotes. A true homozygote for L547V showed LDL and total cholesterol levels distinctly lower than those of true homozygotes for other mutations and similar to those seen in heterozygotes for other mutations. This patient showed no signs of coronary artery disease at 66 years of age, which is unusual for homozygous FH. Family history indicated multiple relatives with mild hypercholesterolemia, consistent with heterozygosity for L547V, and long lifespans in affected family members further supported the mild nature of this mutation. Molecular analyses showed that cell surface LDL binding activity in fibroblasts from the true homozygote for L547V was markedly higher than that observed in homozygotes for other common mutations and similar to that of certain heterozygotes. Internalization and degradation activities were also relatively preserved, and normal synthesis and processing of the LDL receptor protein were observed, indicating that more than half of LDL receptor activity remained on the cell surface. The frequency of the L547V mutation was also examined in compound heterozygotes. Among 12 unrelated compound heterozygotes, L547V was the most frequent mutation, appearing in 25.0% of alleles, a higher frequency than several other common mutations. This contrasted with its lower frequency among FH heterozygotes, suggesting underestimation of its prevalence. Compound heterozygotes with L547V showed slightly lower total and LDL cholesterol levels and higher average LDL binding activity compared with those carrying other mutations. Finally, screening of a general population cohort identified carriers of L547V and c.1845+2T>C mutations, with no carriers of the other common mutations detected, further supporting the likelihood that the frequency of the L547V mutation in FH heterozygote patients has been underestimated.',
    tokenizer
    )

for i, chunk in enumerate(results):
    if i == 0:
        print(i)
        print(len(chunk['text'].split()))
'''
