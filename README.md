# Make-it-clean
Depending on the quality of the original document, Optical Character Recognition (OCR) can produce a range of errors – from erroneous letters to additional and spurious blank spaces. This issue risks to compromise the effectiveness of the analysis tasks in support to the study of texts. Furthermore, the presence of multiple errors of different type in a certain text segment can introduce so much noise that the overall digitization process becomes useless.

The goal of this project is to explore machine learning techniques, in particular sequence to sequence learning, to develop a text correction tool. 

Typically, there are two main errors:

wrong characters in words: (this senlence contains and effor)

wrong segmentation: (th is sentencecont ains an error)

# Dataset
I have used a dataset which consists of 2225 documents from the BBC news website corresponding to stories in five topical areas (business, entertainment, politics, sport, tech) from 2004-2005.

Source: http://mlg.ucd.ie/datasets/bbc.html - D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006

# References
Nastase, V., & Hitschler, J. (2018, May). Correction of OCR word segmentation errors in articles from the ACL collection through neural machine translation methods. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018).

Todorova, K., & Colavizzaa, G. (2020). Transfer Learning for Historical Corpora: An Assessment on Post-OCR Correction and Named Entity Recognition. Proceedings http://ceur-ws. org ISSN, 1613, 0073.

Nguyen, T. T. H., Jatowt, A., Nguyen, N. V., Coustaty, M., & Doucet, A. (2020, August). Neural Machine Translation with BERT for Post-OCR Error Detection and Correction. In Proceedings of the ACM/IEEE Joint Conference on Digital Libraries in 2020 (pp. 333-336).
