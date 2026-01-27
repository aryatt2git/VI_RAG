import weaviate
from embed_model import embed_model
from text2token_chunker import chunk_text_tokens
from transformers import AutoTokenizer, AutoModel
from weaviate.classes.config import Configure, Property, DataType, VectorDistances

class_obj = {
    "class": "ArticleChunk",
    "description": "A chunk of a biological article with embeddings",
    "vectorizer": "none",
    "properties": [
            {
                "name": "title",
                "dataType": ["text"],
                "description": "Title of the article"
            },
            {
                "name": "section",
                "dataType": ["text"],
                "description": "Section of the article"
            },
            {
                "name": "chunk",
                "dataType": ["text"],
                "description": "Chunk of the section from the article"
            },
            {
                "name": "text",
                "dataType": ["text"],
                "description": "Text content of the chunk"
            }
    ]
}

with weaviate.connect_to_custom(
    http_host="localhost",
    http_port=5050,
    http_secure=False,
    grpc_host="localhost",
    grpc_port=50051,
    grpc_secure=False,
) as client:

    if not client.collections.exists("ArticleChunk"):
        client.collections.create(
            name="ArticleChunk",
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                )
            ),
            properties=[
                Property(name="title", data_type=DataType.TEXT),
                Property(name="section", data_type=DataType.TEXT),
                Property(name="chunk", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
            ],
        )

    model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

    dict_rs = {
        'title': 'Update of Japanese common LDLR gene mutations and their phenotypes: Mild type mutation L547V might predominate in the Japanese population',
        'abstract': 'We investigated the LDLR gene mutations in 205 unrelated Japanese FH (familial hypercholesterolemia) heterozygotes to see if there is a prevalence of common mutations in the Japanese population. A total of 53 different small mutations (<25 bp) and 10 kinds of large deletions (>25 bp) were identified. Among them there were eight relatively frequent mutations: C317S, c.1845+2T>C, K790X, L547V, P664L, D412H, c.2312-3C>A and V776M. The patients with these mutations comprised 32% of the total FH heterozygotes investigated. Comparison of clinical phenotypes of the eight frequent mutations disclosed that a missense mutation, L547V, manifested a milder phenotype than the other mutations. The mild clinical phenotype was shown to be based on the high level of receptor activity remaining on the patient’s cell surfaces. When we examined the presence of common mutations in a general population, the L547V mutation was detected with unexpectedly higher frequency than the other mutations, suggesting an underestimation of the frequency of this mutation in FH heterozygote patients. In conclusion, although there is a broad spectrum of LDLR gene mutations in the Japanese population, eight common mutations were observed. Among them, a mild phenotype mutation, L547V, might predominate in the Japanese population.',
        'introduction': 'A mutation in the LDL (low density lipoprotein) receptor (LDLR) gene is known to cause familial hypercholesterolemia (FH). FH is inherited in an autosomal-dominant manner and is one of the most frequent human inherited disorders. The frequency of FH heterozygotes is about one in 500 individuals in most populations in the world. To date, over 1000 different mutations in the LDLR gene have been reported worldwide. With the exception of a small number of founder populations where one or two mutations predominate, most geographically based surveys of FH subjects show a large number of mutations in a given population. We have previously demonstrated the existence of five common mutations in the Japanese population through the screening of mutations found in FH homozygotes in 120 unrelated FH heterozygotes. They were c.1845+2T>C, C317S, K790X, P664L and E119K. In the present study, the precise examination of the LDLR mutation in each subject was conducted in 205 unrelated patients who were clinically diagnosed as FH heterozygotes. Although in general there was a great variety in the LDLR mutations in the Japanese population, we were able to demonstrate eight relatively frequent mutations: C317S, c.1845+2T>C, K790X, L547V, P664L, D412H, c.2312-3C>A and V776M. Through a comparison of the clinical phenotypes of these mutations, a missense mutation, L547V, was shown to manifest a very mild phenotype compared with the other frequent mutations.',
        'methods': 'Two hundred and five unrelated Japanese patients who were clinically diagnosed as FH heterozygotes according to the criteria suggested by the Japan Atherosclerosis Society were investigated in this study. If any of the following diagnostic criteria were satisfied, the proband was diagnosed as an FH heterozygote: type II hyperlipoproteinemia with total cholesterol greater than 260 mg/dl and tendon xanthomas; type II hyperlipoproteinemia with total cholesterol greater than 260 mg/dl and the presence of subjects with type II hyperlipoproteinemia with total cholesterol greater than 260 mg/dl and tendon xanthomas in the proband’s first- or second-degree relatives; or type II hyperlipoproteinemia with total cholesterol greater than 260 mg/dl and LDL receptor activity of the proband’s fibroblasts that was lower than that of a normal control. DNA samples were obtained between 1990 and 2002. Seventy-two percent of the patients investigated were those who visited the lipid clinic of the National Cardiovascular Center, and the rest were those who visited hospitals and universities in various areas of Japan and whose blood samples were sent to us. Sixty-four percent of the patients were born in the Kansai district. Among the total patients, 45.9% were males and 54.1% were females. The mean age was 46.8 years, and 81.8% had tendon or cutaneous xanthomas. The frequency of coronary artery disease in patients over 35 years old was 45.8%. The mean levels of total serum cholesterol, triglycerides, HDL cholesterol, and LDL cholesterol before treatment with lipid-lowering drugs were 367.9 mg/dl, 137.0 mg/dl, 49.6 mg/dl, and 291.7 mg/dl, respectively. There were nine unrelated cases of true homozygotes and 12 unrelated cases of compound heterozygotes for LDLR gene mutations among the patients who visited the lipid clinic of the National Cardiovascular Center or whose blood or fibroblasts were sent from other hospitals between 1990 and 2002. The clinical characteristics of FH homozygotes with different mutations and their LDL receptor activities in fibroblasts were compared. All clinical data and experimental results for the patients were anonymous, and the study protocol was approved by the Ethical Committee of the National Cardiovascular Center. Genomic DNA was isolated from the leukocytes of patients. Point mutations and small insertions and deletions were investigated in the promoter and all coding regions of the LDLR gene using the SSCP method. To amplify the promoter region, exons 1–17, and the coding region of exon 18 in the LDLR gene, reported pairs of primers were used with minor modifications. The amplified products were subjected to SSCP analysis, and fragments showing different patterns from those of normal control subjects were subjected to subcloning followed by sequencing. For samples in which no mutation was found by the SSCP method, direct sequencing of the promoter and all coding regions including exon–intron boundary sequences was performed. For samples in which no mutation was detected by these methods, large deletions or insertions were investigated using a multiplex ligation-dependent probe amplification kit designed to detect deletions or duplications of one or more exons of the LDLR gene. The frequency of eight frequent mutations was determined in participants of a general population cohort using a TaqMan-PCR method. Genomic DNA was isolated from blood leukocytes of the participants, and the study protocol was approved by the Ethical Committee of the National Cardiovascular Center. Fibroblasts were obtained from skin biopsies, and assays of labeled LDL binding, internalization, and degradation were performed at either 4 °C or 37 °C. Activities were expressed as percentages of values obtained in control fibroblasts, and data were obtained from duplicate dishes. LDL receptor protein synthesis was investigated in fibroblasts using established methods.',
        'results': 'Through the examination of the coding and promoter sequences, a small mutation of less than 25 bp in the LDLR gene was identified in 118 patients, representing 57.6% of the 205 unrelated FH heterozygotes. Fifty-three different mutations were identified in total. The nomenclature for the description of sequence variations followed established recommendations. For cDNA sequences, the A of the initiating ATG codon is denoted as nucleotide +1, and amino acids are numbered according to established conventions, with the N-terminal alanine of the mature form numbered as +1. Among the 53 total mutations, 23 had already been reported in Japanese subjects, while the remaining 30 were new Japanese mutations. There was variety in the types of mutations, including missense mutations, nonsense mutations, splicing mutations, deletions, and insertions, with missense mutations comprising 60% of the total. The mutation sites were distributed widely over the entire LDLR gene. There were eight relatively frequent mutations with percentages above 2.4%: C317S, c.1845+2T>C, K790X, L547V, P664L, D412H, c.2313-3C>A, and V776M. These mutations were classified as common mutations. The total number of patients carrying these common mutations was 65, comprising 31.7% of the 205 FH heterozygotes. The remaining 45 mutations were rare, being encountered in only one to three cases, and were present in 25.9% of the patients. For cases in which no mutation was found by these methods, MLPA analysis was performed to detect large DNA rearrangements. At least 10 kinds of deletions were identified, and no duplication or insertion was detected. A total of 21 patients had large deletions, comprising 10.2% of the subjects investigated, a percentage similar to that reported in other studies. Several deletions had been previously reported, while six kinds of deletions had not yet been reported in Japanese subjects. Among the 205 FH heterozygotes, deletions of exon 1 and of exons 2–3 were relatively frequent. At present, the precise deletion sites have not been clarified, so it is not known whether these deletion mutations are uniform, but deletions of exon 1 and exons 2–3 may represent common mutations in Japanese FH heterozygotes. In the remaining 32.2% of patients, no LDLR gene mutation was detected by the present analyses. The clinical features and plasma lipid levels of patients carrying each common LDLR mutation showed no gross differences except for those with the L547V mutation. Heterozygotes with L547V had lower LDL and total cholesterol levels compared with heterozygotes carrying other mutations, while triglyceride and HDL cholesterol levels did not differ significantly except for a lower HDL cholesterol level observed in heterozygotes with C317S. The frequency of xanthomas and the frequency of coronary artery disease in patients over 35 years old were also lower among heterozygotes with L547V, and the frequency of coronary artery disease was significantly lower than in heterozygotes with certain other mutations. These observations indicate that L547V is a mild phenotype mutation. Individual heterozygotes with L547V showed some variability in lipid levels, but average values were lower than those observed in heterozygotes with other mutations, although still higher than in normocholesterolemic individuals, suggesting an elevated coronary artery disease risk compared with the general population. One heterozygote had already developed coronary artery disease by the age of 53. The clinical phenotype of L547V was further compared with other common mutations in true homozygotes. A true homozygote for L547V showed LDL and total cholesterol levels distinctly lower than those of true homozygotes for other mutations and similar to those seen in heterozygotes for other mutations. This patient showed no signs of coronary artery disease at 66 years of age, which is unusual for homozygous FH. Family history indicated multiple relatives with mild hypercholesterolemia, consistent with heterozygosity for L547V, and long lifespans in affected family members further supported the mild nature of this mutation. Molecular analyses showed that cell surface LDL binding activity in fibroblasts from the true homozygote for L547V was markedly higher than that observed in homozygotes for other common mutations and similar to that of certain heterozygotes. Internalization and degradation activities were also relatively preserved, and normal synthesis and processing of the LDL receptor protein were observed, indicating that more than half of LDL receptor activity remained on the cell surface. The frequency of the L547V mutation was also examined in compound heterozygotes. Among 12 unrelated compound heterozygotes, L547V was the most frequent mutation, appearing in 25.0% of alleles, a higher frequency than several other common mutations. This contrasted with its lower frequency among FH heterozygotes, suggesting underestimation of its prevalence. Compound heterozygotes with L547V showed slightly lower total and LDL cholesterol levels and higher average LDL binding activity compared with those carrying other mutations. Finally, screening of a general population cohort identified carriers of L547V and c.1845+2T>C mutations, with no carriers of the other common mutations detected, further supporting the likelihood that the frequency of the L547V mutation in FH heterozygote patients has been underestimated.',
        'discussion': 'In the present study we have shown 19 novel small mutations of less than 25 bp and at least six novel large deletions. Additionally, there were 14 small mutations that had been reported elsewhere in the world but appeared here for the first time as Japanese LDLR gene mutations. Up to now, at least 73 different LDLR gene mutations were known in Japanese FH patients. Our present results bring the number of FH-causing mutations in Japan up to at least 112. These findings demonstrate a broad spectrum of mutations in the LDLR gene in the Japanese population. We have previously shown five relatively frequent LDLR gene mutations, c.1845+2T>C, C317S, K790X, P664L, and E119K, in unrelated FH heterozygotes. As the examination in the present study was performed more precisely and involved a larger number of subjects, the results obtained are believed to better reflect LDLR gene mutations in the Japanese population than previous studies. In the present study, eight mutations were classified as common LDLR mutations, including four previously reported common mutations, with L547V, D412H, c.2312-3C>A, and V776M newly added. In total, these eight common mutations comprised 32% of the FH heterozygotes investigated. There may be regional differences in the mutation spectrum. In some areas, several common mutations together accounted for only a fraction of FH cases, and summarizing these findings, at least four mutations, K790X, P664L, c.1845+2T>C, and c.2312-3C>A, appear to be common across several regions of Japan. However, in each area, common mutations accounted for only about 30–40% of the FH population. Unlike founder populations in which a few mutations explain most FH cases, Japan shows a highly heterogeneous mutation pattern similar to that observed in several European populations. Therefore, genetic diagnosis of LDLR mutations may not be practical in the Japanese population as a whole. In the present analysis, 67.8% of patients clinically diagnosed as FH heterozygotes were attributable to an LDLR gene mutation, while in the remaining 32.2%, known or unknown genetic factors other than the LDLR gene may be involved. Clinical FH can also result from mutations in other genes, such as apolipoprotein B or PCSK9, which may contribute to hypercholesterolemia in some individuals. When the clinical characteristics of the eight common mutations were compared in heterozygotes, patients with the L547V mutation manifested a milder phenotype than those with the other common mutations. Although the range of plasma cholesterol levels within L547V heterozygotes was wide, it was comparable to that observed in heterozygotes with other mutations, suggesting contributions from additional genetic or environmental factors. In the true homozygote for the L547V mutation, LDL and total cholesterol levels were within the range typically observed in heterozygotes for other common mutations. This milder phenotype was attributable to residual LDL receptor activity on the cell surface. Functional studies indicated substantial remaining LDL binding activity in fibroblasts carrying the L547V mutation, consistent with partial preservation of receptor function. The L547V mutation causes a single amino acid substitution within the EGF precursor homology domain of the LDL receptor, which is expected to have only a small effect on receptor structure and function. Because of this very mild phenotype, the frequency of the L547V mutation in FH heterozygotes may be underestimated. Indeed, screening of common mutations in a general population identified carriers of the L547V mutation more frequently than expected. If milder diagnostic criteria were applied, additional subjects with the L547V mutation might be identified. It is also possible that other mild phenotype mutations exist whose carriers are susceptible to misdiagnosis based solely on clinical characteristics.'
    }
    # print(len(dict_rs))

    for key, value in dict_rs.items():
        title = dict_rs['title']

        if key == 'title':
            continue

        section = key
        text = value
        chunks = chunk_text_tokens(title, section, text, tokenizer)

        # encode the queries (use the [CLS] last hidden states as the representations)
        weaviate_embeds = embed_model(chunks, tokenizer, model)

        # View the embeddings by removing the '# '.
        # print(weaviate_embeds)
        # print(weaviate_embeds.shape)

        collection = client.collections.get("ArticleChunk")

        for i, chunk in enumerate(chunks):
            embedding = weaviate_embeds[i].detach().cpu().numpy().tolist()  # make sure it's a numpy array

            # Create a Weaviate object
            collection.data.insert(
                properties={
                    "title": title,
                    "section": section,
                    "chunk": chunk['section'],
                    "text": chunk['text']
                },
                vector=embedding
            )
