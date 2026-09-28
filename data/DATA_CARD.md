# Data card

## Source
Fernandes, K., Cardoso, J., & Fernandes, J. (2017). *Cervical Cancer (Risk Factors)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5Z310

Licence: CC BY 4.0. The file in `data/raw/` is the original, unchanged (SHA-256 `8df193ad5c9ff4288fb4c401eef70dcd2cbda404ce7f82ac74c68cfc960ab063`). The UCI site was not reachable from my build environment, so I took the file from a public mirror and confirmed that three independent copies were byte-for-byte identical, with 858 rows and 36 columns as described on the UCI page.

## Collection
Women attending Hospital Universitario de Caracas, Venezuela. Demographic, sexual, reproductive, smoking, contraceptive and STI history, plus four examination results (Hinselmann, Schiller, cytology, biopsy).

## What I changed
| Step | Detail |
|---|---|
| Missing values | Stored as `?` in the original. Read as missing. Many women declined the sensitive questions, so missing values come in blocks. |
| Duplicates | 23 exact duplicate rows removed before splitting (858 to 835). |
| Outcome | `Biopsy` (1 = positive). 54 positives, 6.5%. |
| Excluded for leakage | Hinselmann, Schiller, Citology (same work-up as the biopsy); Dx:Cancer, Dx:CIN, Dx:HPV, Dx (existing diagnoses, handled by the rules layer). |
| Excluded for quality | STDs: Time since first diagnosis, STDs: Time since last diagnosis (92% missing). |
| Kept | 13 history-taking features, listed in `src/cervical_cdss/features.py`. |
| Split | 75% development (626 women, 40 positive), 25% locked test (209 women, 14 positive), stratified, seed 42. |

## Things to know
- The women are young: the median age is 26, two thirds are under 30, and only 8 are over 50.
- No HPV test results are included.
- It is not stated whether every woman had a biopsy, so some negatives may be unverified.
- Files in `data/sample/` are made up for demonstrating the batch upload. They are not real patients.
