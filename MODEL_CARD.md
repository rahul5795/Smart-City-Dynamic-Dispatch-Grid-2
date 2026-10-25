# Model card: Sahayak triage model (2026-10-25)

**What it does.** Reads one emergency-call transcript and predicts the incident type (collapse, fire, gas leak, flood, medical, or noise)
and the true severity (1-5), each with a calibrated probability.

**Intended use.** Decision support in a training simulation. A human dispatcher stays in charge; calls below
77% confidence are sent for human review. Not for real emergencies.

**Data.** Synthetic calls generated from an answer key (1354 development calls from 30 earthquake scenarios;
927 hidden-test calls from 20 flood scenarios). Labels are the true incident facts, not what callers said.

**Model.** Character n-grams (2-5, within words) with sublinear TF-IDF, multinomial logistic regression with balanced class weights.
C chosen by 5-fold group cross-validation (groups = scenarios): type C=0.3, severity C=1.0. Probabilities calibrated by
temperature scaling on out-of-fold development predictions (type T=0.05, severity T=0.62).

**Safety policy.** On development data the **model was** better at the incident type (macro-F1: rules 0.885,
model 0.999 out-of-fold), so it leads; the other side only takes over above the review threshold. The model may
rescue a call the rules dismissed but may never dismiss a call the rules accepted. Severity is the higher of rules and model,
so the model can raise urgency but never lower it. When rules and model disagree, the call is flagged for a human.

**Results on the hidden flood test (scored once).**
| Metric | Rules | Safety hybrid |
|---|---|---|
| Type macro-F1 (model alone: 1.000) | 0.928 | 1.000 |
| Noise filtered | 100.0% | 100.0% |
| Real calls ignored | 0.1% | 0.0% |
| Severity within 1 | 97.9% | 97.7% |
| Severe incidents under-triaged | 14.4% | 11.6% |
| Harm-weighted cost per call | 0.825 | 0.492 |

Expected calibration error on the hidden test: 0.416 before temperature scaling, 0.000 after.

**Known weakness.** Near-perfect scores here mostly mean the model has learned the call generator's phrasing. Real callers, accents and speech-to-text errors will be harder, so expect lower accuracy and re-calibrate on real data before trusting the numbers.

**Limitations.** Trained only on synthetic English and Hinglish text from one generator, so real callers, other languages and
speech-to-text errors will be harder. Severity labels come from the answer key, which simplifies real clinical judgement.
Re-check calibration whenever the data changes.
