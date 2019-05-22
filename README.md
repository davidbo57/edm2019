# Educational Data Mining (2019) - Poster
Boulanger, D., & Kumar V. (2019, accepted). Shedding Light on the Automated Essay Scoring Process. In Proceedings of the 2019 International Conference on Educational Data Mining (EDM), Poster, July 2-5, Montreal, Canada.

## Automated Student Assessment Prize
The ASAP's essay datatsets are available at https://www.kaggle.com/c/asap-aes.

## Suite of Automatic Linguistic Analysis Tools
For more information on the software tools used to extract the writing features used in this paper, please visit https://www.linguisticanalysistools.org/. These tools are also described in depth in the following papers:

Crossley, S.A., Kyle, K., Davenport, J.L. and McNamara, D.S., 2016, June. Automatic Assessment of Constructed Response Data in a Chemistry Tutor. In EDM (pp. 336-340). Tavel, P. 2007. Modeling and Simulation Design.

Crossley, S.A., Kyle, K. and McNamara, D.S., 2016. The tool for the automatic analysis of text cohesion (TAACO): Automatic assessment of local, global, and text cohesion. Behavior research methods, 48(4), pp.1227-1237.

Kyle, K., Crossley, S. and Berger, C., 2018. The tool for the automatic analysis of lexical sophistication (TAALES): version 2.0. Behavior research methods, 50(3), pp.1030-1046.

Kyle, K., 2016. Measuring syntactic development in L2 writing: Fine grained indices of syntactic complexity and usage-based indices of syntactic sophistication.

Lu, X., 2010. Automatic analysis of syntactic complexity in second language writing. International journal of corpus linguistics, 15(4), pp.474-496.

## The 96 Most Correlated (Spearman) Writing Features with Essay Holistic Scores

Rank|Tool|Spearman's Rho|Writing Feature ID|Writing Feature Description|Category
---|---|---:|---|---|---
1|TAALES|0.525|LD_Mean_RT_Zscore_CW|Lexical Decision Time (z-score) CW|Word Recognition Norms
2|TAALES|-0.508|PLDF_CW|Average log HAL frequency of closest phonological neighbors CW|Word Neighbor Information
3|TAALES|0.505|LD_Mean_RT_CW|Lexical Decision Time CW|Word Recognition Norms
4|TAALES|-0.492|OLDF_CW|Average log HAL frequency of closest orthographic neighbors CW|Word Neighbor Information
5|TAALES|0.479|WN_Zscore_CW|Word Naming Response Time (z-score) CW|Word Recognition Norms
6|TAALES|0.470|WN_Zscore|Word Naming Response Time (z-score)|Word Recognition Norms
7|TAALES|-0.469|Ortho_N_CW|Orthographic Neighbors CW|Word Neighbor Information
8|TAALES|0.463|WN_Mean_RT|Word Naming Response Time|Word Recognition Norms
9|TAALES|0.461|LD_Mean_RT_Zscore|Lexical Decision Time (z-score)|Word Recognition Norms
10|TAALES|-0.461|SUBTLEXus_Freq_AW_Log|SUBTLEXus Frequency AW Logarithm|Word Frequency
11|TAALES|0.459|PLD_CW|Average Levenshtein Distance of closest phonological neighbors CW|Word Neighbor Information
12|TAALES|-0.456|SUBTLEXus_Range_AW|SUBTLEXus Range AW|Word Range
13|TAALES|-0.456|OLDF|Average log HAL frequency of closest orthographic neighbors|Word Neighbor Information
14|TAALES|0.452|PLD|Average Levenshtein Distance of closest phonological neighbors|Word Neighbor Information
15|TAALES|0.451|WN_Mean_RT_CW|Word Naming Response Time CW|Word Recognition Norms
16|TAALES|-0.450|SUBTLEXus_Freq_CW_Log|SUBTLEXus Frequency CW Logarithm|Word Frequency
17|TAALES|0.447|LD_Mean_RT|Lexical Decision Time|Word Recognition Norms
18|TAALES|-0.445|PLDF|Average log HAL frequency of closest phonological neighbors|Word Neighbor Information
19|TAALES|0.444|OLD_CW|Average Levenshtein Distance of closest orthographic neighbors CW|Word Neighbor Information
20|TAALES|0.443|Kuperman_AoA_CW|Age of Acquisition CW|Age of Acquisition/Exposure
21|TAALES|-0.439|Phono_N|Phonological Neighbors (homonyms excluded)|Word Neighbor Information
22|TAALES|-0.439|Phono_N_H|Phonological Neighbors (homonyms included)|Word Neighbor Information
23|TAALES|-0.435|BNC_Spoken_Freq_AW_Log|BNC Spoken Frequency AW Logarithm|Word Frequency
24|TAALES|-0.435|COCA_spoken_Frequency_Log_AW|COCA Spoken Frequency AW Logarithm|Word Frequency
25|TAALES|-0.435|Phono_N_CW|Phonological Neighbors (excludes homonyms) CW|Word Neighbor Information
26|TAALES|-0.434|Brown_Freq_AW_Log|Brown Frequency AW Logarithm|Word Frequency
27|TAALES|-0.434|TL_Freq_AW_Log|Thorndike-Lorge Frequency AW Logarithm|Word Frequency
28|TAALES|-0.432|COCA_spoken_Range_AW|COCA Spoken Range AW|Word Range
29|CRAT|0.432|nwords|Number of words|
30|TAALES|-0.432|Phono_N_H_CW|Phonological Neighbors (includes homonyms) CW|Word Neighbor Information
31|TAALES|-0.431|SUBTLEXus_Range_CW|SUBTLEXus Range CW|Word Range
32|TAALES|-0.427|Log_Freq_HAL|HAL Frequency Logarithm|Word Frequency
33|TAALES|-0.424|Ortho_N|Orthographic Neighbors|Word Neighbor Information
34|TAALES|0.423|Kuperman_AoA_AW|Age of Acquisition AW|Age of Acquisition/Exposure
35|TAALES|-0.421|Brown_Freq_CW_Log|Brown Frequency CW Logarithm|Word Frequency
36|TAALES|-0.414|BNC_Spoken_Range_AW|BNC Spoken Range AW|Word Range
37|TAALES|-0.414|SUBTLEXus_Range_AW_Log|SUBTLEXus Range AW Logarithm|Word Range
38|TAALES|-0.410|COCA_fiction_Range_AW|COCA Fiction Range AW|Word Range
39|TAALES|-0.409|TL_Freq_CW_Log|Thorndike-Lorge Frequency CW Logarithm|Word Frequency
40|TAALES|-0.405|BNC_Spoken_Freq_CW_Log|BNC Spoken Frequency CW Logarithm|Word Frequency
41|TAALES|0.395|OLD|Average Levenshtein Distance of closest orthographic neighbors|Word Neighbor Information
42|TAALES|-0.392|SUBTLEXus_Range_CW_Log|SUBTLEXus Range CW Logarithm|Word Range
43|TAALES|0.391|McD_CD_CW|McDonald Co-occurrence Probability CW|Contextual Distinctiveness
44|TAALES|0.390|COCA_fiction_tri_2_DP|COCA Fiction Trigram Bigram to Unigram Association Strength (DP)|Ngram Association Strength
45|TAALES|-0.390|COCA_spoken_Range_Log_AW|COCA Spoken Range AW Logarithm|Word Range
46|TAALES|-0.390|OG_N|Phonographic Neighbors (homophones excluded)|Word Neighbor Information
47|TAALES|-0.389|OG_N_CW|Phonographic Neighbors (homophones excluded) CW|Word Neighbor Information
48|TAALES|-0.386|Log_Freq_HAL_CW|HAL Frequency Logarithm CW|Word Frequency
49|TAALES|-0.386|SUBTLEXus_Freq_CW|SUBTLEXus Frequency CW|Word Frequency
50|TAALES|-0.380|BNC_Spoken_Range_CW|BNC Spoken Range CW|Word Range
51|TAALES|-0.380|COCA_news_Range_AW|COCA News Range AW|Word Range
52|TAALES|-0.379|COCA_spoken_Frequency_Log_CW|COCA Spoken Frequency CW Logarithm|Word Frequency
53|TAALES|-0.378|COCA_spoken_Range_CW|COCA Spoken Range CW|Word Range
54|TAALES|-0.375|COCA_fiction_Frequency_Log_AW|COCA Fiction Frequency AW Logarithm|Word Frequency
55|TAALES|-0.371|MRC_Familiarity_AW|MRC Familiarity AW|Psycholinguistic Norms
56|TAALES|-0.369|COCA_magazine_Frequency_Log_AW|COCA Magazine Frequency AW Logarithm|Word Frequency
57|TAACO|-0.367|adjacent_overlap_cw_sent|adjacent sentence overlap content lemmas|Lexical Overlap (Sentence)
58|TAALES|-0.366|COCA_news_Frequency_Log_AW|COCA News Frequency AW Logarithm|Word Frequency
59|TAALES|0.366|COCA_magazine_tri_2_DP|COCA Magazine Trigram Bigram to Unigram Association Strength (DP)|Ngram Association Strength
60|TAALES|-0.363|COCA_fiction_Range_CW|COCA Fiction Range CW|Word Range
61|TAALES|-0.363|BNC_Spoken_Freq_CW|BNC Spoken Frequency AW|Word Frequency
62|TAALES|0.359|aoe_inflection_point_polynomial|LDA Age of Exposure (inflection point)|Age of Acquisition/Exposure
63|TAACO|-0.359|adjacent_overlap_all_sent|adjacent sentence overlap all lemmas|Lexical Overlap (Sentence)
64|TAALES|-0.357|COCA_fiction_Frequency_Log_CW|COCA Fiction Frequency CW Logarithm|Word Frequency
65|TAALES|0.355|LD_Mean_RT_SD_CW|Lexical Decision Time (standard deviation) CW|Word Recognition Norms
66|TAASSC|-0.355|fic_av_lemma_freq_log|average lemma frequency, log transformed - fiction|Syntactic Sophistication
67|TAALES|-0.353|COCA_news_Range_Log_AW|COCA News Range AW Logarithm|Word Range
68|TAALES|-0.353|Brown_Freq_CW|Brown Frequency AW|Word Frequency
69|TAALES|-0.352|COCA_magazine_Range_AW|COCA Magazine Range AW|Word Range
70|TAACO|-0.351|adjacent_overlap_2_cw_sent|adjacent two-sentence overlap content lemmas|Lexical Overlap (Sentence)
71|TAALES|-0.350|KF_Nsamp_AW|Kucera-Francis Range AW|Word Range
72|TAALES|-0.349|KF_Freq_CW_Log|Kucera-Francis Frequency CW Logarithm|Word Frequency
73|TAASSC|-0.345|news_av_lemma_freq_log|average lemma frequency, log transformed - news|Syntactic Sophistication
74|TAALES|-0.344|COCA_fiction_Range_Log_AW|COCA Fiction Range AW Logarithm|Word Range
75|TAALES|-0.343|USF_CW|Free Association Stimuli Elicited CW|Contextual Distinctiveness
76|TAACO|-0.342|prp_ttr|pronoun lemma TTR|TTR and Density
77|TAALES|0.342|WN_SD|Word Naming Response Time (standard deviation)|Word Recognition Norms
78|TAALES|-0.340|COCA_magazine_Range_Log_AW|COCA Magazine Range AW Logarithm|Word Range
79|TAALES|-0.340|MRC_Familiarity_CW|MRC Familiarity CW|Psycholinguistic Norms
80|TAALES|0.340|aoe_inverse_average|LDA Age of Exposure (inverse average)|Age of Acquisition/Exposure
81|TAALES|-0.339|COCA_spoken_Range_Log_CW|COCA Spoken Range CW Logarithm|Word Range
82|TAALES|0.339|COCA_news_tri_2_DP|COCA News Trigram Bigram to Unigram Association Strength (DP)|Ngram Association Strength
83|TAALES|-0.339|KF_Freq_AW_Log|Kucera-Francis Frequency AW Logarithm|Word Frequency
84|TAACO|-0.338|adjacent_overlap_cw_sent_div_seg|adjacent sentence overlap content lemmas (sentence normed)|Lexical Overlap (Sentence)
85|TAALES|-0.337|COCA_news_Frequency_Log_CW|COCA News Frequency CW Logarithm|Word Frequency
86|TAACO|-0.335|adjacent_overlap_2_all_sent|adjacent two-sentence overlap all lemmas|Lexical Overlap (Sentence)
87|TAALES|-0.334|COCA_news_Range_CW|COCA News Range CW|Word Range
88|TAASSC|-0.333|mag_av_lemma_freq_log|average lemma frequency, log transformed - magazine|Syntactic Sophistication
89|SEANCE|0.330|Trngain_Lasswell_neg_3||Sentiment Analysis
90|TAASSC|0.328|av_nsubj_pass_deps_NN|dependents per passive nominal subject (no pronouns)|Noun Phrase Complexity
91|TAALES|0.327|COCA_fiction_bi_MI|COCA Fiction Bigram Association Strength (MI)|Ngram Association Strength
92|TAALES|0.322|aoe_index_above_threshold_40|LDA Age of Exposure (.40 cosine threshold)|Age of Acquisition/Exposure
93|SEANCE|0.322|Trngain_Lasswell||Sentiment Analysis
94|TAASSC|0.322|nsubj_pass_stdev|dependents per passive nominal subject (standard deviation)|Noun Phrase Variety
95|TAALES|-0.322|COCA_magazine_Frequency_Log_CW|COCA Magazine Frequency CW Logarithm|Word Frequency
96|TAALES|-0.320|KF_Nsamp_CW|Kucera-Francis Range CW|Word Range