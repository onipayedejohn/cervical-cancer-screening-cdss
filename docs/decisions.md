# Design decisions and questions to be ready for

Notes for myself: the reasoning behind each choice, and the questions a reviewer is likely to ask.

**Why not use Hinselmann, Schiller and cytology?**
They are examination results from the same visit as the biopsy. At the time a screening decision is made they do not exist, so using them is leakage. Including them raises test ROC-AUC from 0.67 to 0.90.

**Why exclude the Dx columns?**
They record existing diagnoses (cancer, CIN, HPV). A woman with a known diagnosis is not in routine screening, so the rules layer handles her instead of the model.

**Why remove duplicates before splitting?**
If one record is in both training and test sets, the test score measures memory, not generalisation.

**Why PR-AUC and not accuracy?**
Only about 6% of cases are positive. Predicting "negative" for everyone gives 94% accuracy and is useless. PR-AUC focuses on how well the positives are found; its no-skill value equals the prevalence (0.064).

**Why nested cross-validation?**
If you tune hyperparameters and report the score from the same folds, the score is optimistic. The inner loop tunes, the outer loop measures.

**Why is the threshold chosen for 80% sensitivity?**
In screening, missing a lesion is worse than an extra follow-up conversation. The threshold was picked on out-of-fold development predictions, never on the test set.

**Why calibrate?**
Class weighting pushes predicted probabilities up. Sigmoid calibration brings them back to real rates, so "6%" means roughly 6 in 100.

**Why gradient boosting?**
Highest mean PR-AUC in nested CV. The pre-set tie-break (within 0.01, pick the greener model) did not apply because the next model was 0.01 lower.

**How was energy measured?**
CodeCarbon's offline tracker with Ghana's grid carbon intensity. On a cloud VM without power counters it estimates CPU power from the processor type, so compare models against each other, not the absolute values.

**What is the biggest limitation?**
Very few women over 50 and no HPV results, from a single hospital in Caracas. I would want external validation on a Ghanaian cohort, ideally with HPV results, before taking any number seriously.

**What would I do next?**
1. Test the rules with a clinician and compare them with Ghana's national screening guidance.
2. Deliver the rules as a SMART on FHIR app or CDS Hooks service so they can sit inside an EHR.
3. Replace the model's inputs with HPV genotype and laboratory data when a suitable dataset is available.
