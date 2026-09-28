# Model card: cervical biopsy risk estimate

## Intended use
Adds context to a guideline-based screening decision in a teaching prototype. It estimates the chance that a woman would have a positive cervical biopsy, using only questions asked at a screening visit.

**Not intended for:** deciding whether someone should be screened, replacing an HPV test, VIA or cytology, or any real clinical decision. The WHO rules layer makes the screening decision and the model cannot change it.

## Model
- Gradient boosting (scikit-learn `HistGradientBoostingClassifier`, class-weighted), learning rate 0.1, max depth 3, 100 iterations.
- Median imputation with missing-value indicators inside the pipeline.
- Sigmoid calibration with 5-fold `CalibratedClassifierCV`.
- 13 inputs: age, lifetime sexual partners, age at first sex, pregnancies, years smoking, pack-years, years on hormonal contraception, years with an IUD, number of past STIs, genital warts, syphilis, HIV, number of STI diagnoses.

## How it was chosen
Five candidates (no-skill baseline, logistic regression, random forest, gradient boosting, RBF SVM) were compared with nested cross-validation on the development set only: 5 folds × 5 repeats outside, 3-fold grid search inside, scored by PR-AUC. Rule fixed in advance: highest mean PR-AUC; within 0.01, the lower-energy model wins.

## Operating point
Threshold 0.056, the highest cut-off that reached 80% sensitivity on out-of-fold development predictions. Risk bands use the 40th and 80th percentiles of those predictions (0.061 and 0.079).

## Performance on the locked test set (209 women, 14 positive)
| Metric | Value | 95% bootstrap CI |
|---|---|---|
| ROC-AUC | 0.67 | 0.51 to 0.80 |
| PR-AUC | 0.12 (no skill 0.07) | 0.06 to 0.23 |
| Sensitivity | 0.86 | 0.64 to 1.00 |
| Specificity | 0.34 | 0.28 to 0.41 |
| PPV | 0.09 | 0.04 to 0.14 |
| NPV | 0.97 | 0.93 to 1.00 |
| Brier score | 0.062 | 0.036 to 0.093 |

Observed positive rate by band: lower 2.2%, average 7.8%, higher 12.5%.

**Plain reading:** the model sorts women into lower and higher risk groups better than chance, but it is weak on its own. At the chosen threshold it flags about two thirds of women, so it could not work as a stand-alone triage tool.

## Subgroups (out-of-fold, development set)
| Age | Women | Positives | ROC-AUC | Sensitivity |
|---|---|---|---|---|
| Under 30 | 414 | 25 | 0.58 | 0.76 |
| 30 to 49 | 206 | 13 | 0.59 | 0.85 |
| 50 and over | 6 | 2 | not meaningful | not meaningful |

Too few older women to say anything about them. This is the most important gap for a screening tool, since screening targets women aged 30 and over.

## What the model relies on
Permutation importance (on the development set, so it shows what the fitted model uses, not what generalises): years on hormonal contraception, age at first sex, age, lifetime partners and pregnancies. These are associations in one dataset, not causes.

## Leakage check
With the colposcopy, Schiller, cytology and prior-diagnosis columns added back, a random forest reaches a test ROC-AUC of 0.90 and PR-AUC of 0.69. That gap shows how much of the high performance reported for this dataset comes from information that is not available at the time of screening.

## Environmental cost
Measured with CodeCarbon (Ghana grid intensity). The full nested cross-validation for gradient boosting used about 0.08 Wh; the random forest used about 0.7 Wh for no gain. Scoring 1,000 women takes about 2 ms. Values are estimates on a cloud CPU and vary between runs.

## Ethical considerations
- Trained on women from one Venezuelan hospital. Transfer to Ghana, or anywhere else, is unknown.
- Sensitive questions have many missing answers. The model treats "did not answer" as information, which could reflect social factors rather than biology.
- A risk estimate on sexual history can feel judgemental to a patient. The app frames it as context for the clinician, not a label for the woman.

## Versions
scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4. Trained with seed 42. Full details in `models/model_metadata.json`.
