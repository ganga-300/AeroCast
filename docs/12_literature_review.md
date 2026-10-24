# Literature Review (verified sources) — basis for the IEEE report

**Status:** draft for the Related Work / Background sections of the Phase 1 IEEE report.
BibTeX: `report/references.bib`; keys are shown as `[key]` below. Sources checked on 2026-10-24.

## 0. How this review was built (and its limits)
This is a **focused, non-systematic review**. It covers (A) flight-delay prediction studies, compared
on the dimensions that matter for this project, and (B) methodological references that justify
specific AeroCast design choices. Every entry was checked against an authoritative record.
Verification levels:

| level | meaning |
|---|---|
| **F** | full text (or the relevant sections) read |
| **A** | abstract read on the publisher, index or author page |
| **M** | bibliographic metadata verified (Crossref, CRAN, PMLR, publisher page); claims limited to what the title or abstract-level summaries state |
| **S** | detail taken from another verified paper's summary of it; attributed explicitly in the text |

**What I tried and couldn't use:**
- Full texts behind paywalls or anti-bot pages (ACM Digital Library, ScienceDirect) were not
  read, and no access check was bypassed.
- One TU Delft repository file failed to download.
- Candidate papers whose records could not be confirmed were dropped, e.g. Yu et al. (2019) and
  Kalliguddi & Leboulluec (2017), which are cited in [zoutendijk2021] but were not verified here.

Do not cite anything in the report beyond the level recorded for it.

## 1. Data provenance and target definition
- **BTS definition** [bts_ontime] (F, primary page). A flight counts as *on time* if it operated
  **less than 15 minutes** after the scheduled time in the carriers' reservation systems.
  - Arrival performance is measured at the gate.
  - Reporting covers domestic operations of carriers with at least 0.5% of domestic scheduled
    passenger revenue.
  - Carriers report delay causes in five categories (air carrier, extreme weather, National
    Aviation System, late-arriving aircraft, security).

  AeroCast's target, `arr_delay ≥ 15` minutes, matches this convention (D6).
- **Dataset** [nycflights13] (F, package documentation). `nycflights13` contains on-time data for
  all flights that departed NYC (JFK, LGA, EWR) in 2013. It is documented as sourced from
  RITA / Bureau of Transportation Statistics, and its README states 336,776 flights. This equals the
  row count of the supplied Kaggle file (Stage 1 audit), which supports the provenance statement in
  `docs/01`.

## 2. Flight-delay prediction literature (group A)
### 2.1 Thematic synthesis
**Surveys.** [carvalho2021] (A; preprint [sternberg2017]) reviews data-science methods for flight
delays. It proposes a taxonomy organised by problem scope, data and computational method, and
traces trends in the field. Its own conclusions (counts, prevalence of methods) were not read in
full here.

**Short-horizon models that use the current state of the airspace.**
- [rebollo2014] (A): Random Forests predicting departure delays **2–24 h ahead**. The features
  combine temporal and network delay states, including network-wide variables describing the delay
  condition of the US National Airspace System *at the time of prediction*. The work is trained and
  validated on 2007–2008 data, with test results reported both for classification at a threshold
  and for delay values.
- [gui2020] (A): ADS-B surveillance messages combined with weather, schedule and airport data.
  Compares learners including LSTM and random forest; reports overfitting for the LSTM on their
  limited dataset and 90.2% binary accuracy for random forest.
- [kim2016] (A): models day-to-day sequences of an airport's departure and arrival delays with deep
  LSTM recurrent networks, reporting improved accuracy over earlier methods.

These designs rely on information that exists only close to departure (live network state,
surveillance data, recent delay sequences).

**Longer horizons with schedule or forecast inputs.**
- [choi2016] (A): uses US domestic flight and weather data (2005–2015). At prediction time it feeds
  **flight schedule and weather forecast** data into binary classifiers (decision trees, random
  forest, AdaBoost, k-nearest neighbours). It addresses class imbalance with sampling and compares
  models by accuracy and ROC curves. According to [zoutendijk2021] (S), it considers horizons of
  **five days and one day** and reports 0.80 accuracy for random forest.
- [lambelho2020] (A) evaluates **strategic** schedules at London Heathrow (2013–2018), where slots
  are assigned months ahead, using machine-learning predictions of delays and cancellations. The
  abstract does not state the learners, features or validation design. [zoutendijk2021] (S) reports
  that it used LightGBM, a multilayer perceptron and random forests, with an average F1 of 0.56.

**Imbalance and costs.** [choi2017] (A) makes classifiers cost-sensitive by subsampling according to
misclassification costs. It evaluates a weighted error that depends on the cost ratio between false
positives and false negatives, noting that the best setting depends on that ratio.

**Probabilistic prediction.** [zoutendijk2021] (F, sections 1–2.6) predicts **delay distributions**
for individual flights at Rotterdam The Hague Airport (Jan 2017–Feb 2020):
- **Inputs:** schedule features plus weather measured at origin and destination airports.
- **Models:** mixture density networks and random-forest regression with kernel density estimation.
- **Scoring:** the CRPS, a proper scoring rule, alongside RMSE/MAE variants and spread metrics.
  Reported MAE is under 15 minutes.
- **Validation:** train and test sets are built with **5-fold cross-validation**.
- **Positioning:** the paper argues that most prior work reports classes or point estimates with
  confusion-matrix or RMSE/MAE metrics, so it says little about uncertainty.

### 2.2 Comparison table
"n.s." means not stated in the parts read; it does **not** mean the study lacks the property.

| study | question | data | horizon | information assumed at prediction time | model | validation | metrics | limitation / relevance to AeroCast | ver. |
|---|---|---|---|---|---|---|---|---|---|
| [rebollo2014] | departure delay class/value for OD links | US NAS operations 2007–08; 100 most-delayed links | 2–24 h | temporal + network delay state at prediction time | Random Forest | trained/validated on 2007–08; split n.s. | test error (60-min threshold), median delay error | live network state unavailable a day ahead; shows the value of current-state features AeroCast cannot use | A |
| [choi2016] | weather-induced delay (binary) | US domestic flights + weather 2005–15 | 1 and 5 days (S) | schedule + **weather forecast** | DT, RF, AdaBoost, kNN | n.s. | accuracy, ROC | closest horizon to AeroCast; relies on weather forecasts (excluded by D8); accuracy under imbalance | A/S |
| [choi2017] | cost-sensitive delay classification | n.s. | n.s. | n.s. | costing + supervised learners | n.s. | cost-weighted error | supports treating error costs and imbalance explicitly | A |
| [kim2016] | airport delay prediction | n.s. | n.s. (several hours per [zoutendijk2021], S) | day-to-day delay sequences | deep LSTM RNN | n.s. | accuracy | sequence models are the Phase 2 direction; accuracy-centred | A |
| [gui2020] | generalised delay classification/regression | ADS-B + weather + schedule + airport | n.s. | surveillance + weather | LSTM, RF, others | n.s. | accuracy (90.2% binary) | real-time surveillance inputs; accuracy reporting | A |
| [lambelho2020] | strategic schedule assessment via delay/cancellation prediction | Heathrow 2013–18 | strategic (months-ahead slots); exact horizon n.s. | n.s. | LightGBM, MLP, RF (S) | n.s. | F1 0.56 (S) | strategic horizon is closest in spirit; details unverified | A/S |
| [zoutendijk2021] | per-flight delay **distribution** | Rotterdam 2017–Feb 2020 | n.s. in sections read | schedule + measured origin/destination weather | MDN, RF + KDE | 5-fold CV | CRPS (proper), RMSE_M, MAE_M, spread | probabilistic evaluation like AeroCast's calibration focus; random CV, not chronological | F |
| [carvalho2021] | survey/taxonomy of delay research | literature | — | — | — | — | — | context and taxonomy | A |

## 3. Methodological foundations (group B)
| AeroCast choice | source(s) | what the source supports | ver. |
|---|---|---|---|
| Gradient boosting as the nonlinear model | [friedman2001], [chen2016] | Friedman formulates boosting as greedy function approximation. Chen & Guestrin describe XGBoost as a scalable tree-boosting system with a **sparsity-aware** algorithm that handles missing values directly, plus a weighted quantile sketch | M / A |
| Trees for tabular data; deep learning deferred to Phase 2 | [grinsztajn2022] | on a 45-dataset tabular benchmark, tree-based models remain state of the art at medium sizes; relevant to choosing XGBoost now | A |
| Logistic Regression with L2 regularisation | [hastie2009] | standard reference for logistic regression and regularisation | M |
| AP as primary metric; ROC as secondary | [saito2015], [davis2006] | Saito & Rehmsmeier show ROC plots can mislead on imbalanced data, while precision–recall plots reflect the share of true positives among positive predictions. Davis & Goadrich study the relationship between PR and ROC curves | A / M |
| Brier score and log loss | [brier1950], [gneiting2007] | Brier introduced the probability score now named after him; Gneiting & Raftery treat strictly proper scoring rules | M |
| Calibration methods (sigmoid, isotonic) | [platt2000], [zadrozny2002], [niculescumizil2005] | Platt maps classifier outputs to probabilities with a fitted sigmoid. Zadrozny & Elkan transform scores into calibrated probabilities. Niculescu-Mizil & Caruana find that boosted trees distort probabilities towards the middle and compare Platt scaling with isotonic regression, including their data needs | M / M / A |
| Calibration matters and is assessed explicitly | [guo2017], [vancalster2019] | Guo et al. show modern neural networks are often poorly calibrated, with temperature scaling (a one-parameter Platt variant) effective. Van Calster et al. argue poor calibration misleads decisions, and that calibration should be checked at validation and models updated when needed | A / A |
| Leakage prevention (contract, D−2 cutoff, learn–predict separation) | [kaufman2012] | defines leakage as information about the target that would not legitimately be available, and recommends a learn–predict separation | A |
| Temporal validation | [bergmeir2012] | warns that dependence in time series violates cross-validation's assumptions in theory. **Finds no practical harm in its experiments and recommends blocked cross-validation**, so it does not, by itself, condemn random splits | A |
| Interpreting the December deterioration | [morenotorres2012], [saerens2002] | Moreno-Torres et al. unify the terminology of dataset shift. Saerens et al. adjust classifier outputs to new class priors, the basis of the Stage 9 prior-shift check | A / M |
| Day-level bootstrap intervals | [field2007] | studies bootstrap methods for clustered data. Which bootstrap is consistent depends on the assumed model; the cluster bootstrap is consistent under both models they consider | A |

## 4. Synthesis: what the reviewed work does and does not establish
1. **Horizon and information availability are often implicit.** The studies with stated horizons
   cover very different ones: 2–24 h [rebollo2014], 1–5 days [choi2016] (S), and strategic months
   [lambelho2020]. The strongest short-horizon results use information that only exists near
   departure: live network state, surveillance data, recent delay sequences. Results from different
   horizons are therefore not comparable. A model claiming a day-ahead horizon must exclude such
   inputs, as AeroCast's contract does.
2. **Evaluation is commonly accuracy-centred.** In the abstracts reviewed, accuracy or
   confusion-matrix metrics are the reported headline [choi2016], [kim2016], [gui2020]. Under
   imbalance this can mislead [saito2015]; AeroCast's majority-class dummy scores 84% "accuracy" on
   validation.
3. **Probabilistic quality is rarely evaluated.** [zoutendijk2021] is the clear exception in this
   sample, and the calibration literature [guo2017], [vancalster2019], [niculescumizil2005] shows
   why it matters.
4. **Validation design varies and is often not stated in abstracts.** One fully read study uses
   random 5-fold CV [zoutendijk2021]. Temporal dependence makes the choice consequential
   [bergmeir2012], although cross-validation need not always fail.
5. **Leakage is a recognised failure mode** [kaufman2012]. Historical aggregates are a subtle route,
   when outcomes not yet known at prediction time are included.

## 5. Gap statement and positioning of AeroCast
Within the verified sample, no study combines **all** of:
- (i) an explicit day-before prediction contract, with historical outcomes admitted only if they
  were verifiably complete before the prediction time (D−2, with landing-time evidence);
- (ii) a chronological train/validation/test design with a single, pre-registered,
  access-logged test evaluation;
- (iii) imbalance-aware ranking metrics (AP) alongside proper scoring rules and calibration
  assessed *across time*;
- (iv) explicit reporting of a negative operational result: no 70%-precision operating point, and
  calibration deteriorating under seasonal shift.

AeroCast is positioned as a **methodologically rigorous, leakage-controlled baseline study** for
day-ahead delay risk. It is not a state-of-the-art accuracy claim. Its contribution is showing
what a strictly day-ahead model can and cannot deliver when evaluated honestly.

**Caveat for the report:** the review is small and non-systematic, and several entries were checked
at abstract level only. The gap statement must be phrased as "in the reviewed literature", not as a
claim about the whole field.

## 6. How each section feeds the IEEE report
- **Related Work:** §2.1 and the §2.2 table, condensed to about 6 studies.
- **Background / Method:** §1 for data and target; §3 for model, metric and calibration
  justifications.
- **Discussion:** §4–5, connected to `docs/10` and `docs/11` results.
- **Limitations:** §0 and the §5 caveat.

## 7. Verification log
| key | record checked | level |
|---|---|---|
| bts_ontime | bts.gov page (browser) | F |
| nycflights13 | CRAN page; GitHub README; `R/flights.R` @source | F |
| carvalho2021 / sternberg2017 | IDEAS/RePEc metadata; arXiv abstract | A |
| rebollo2014 | MIT DSpace record (abstract) | A |
| choi2016 | Crossref; IEEE Xplore abstract (browser) | A |
| choi2017 | Crossref; IEEE Xplore abstract | A |
| kim2016 | Crossref; IEEE Xplore abstract | A |
| gui2020 | Crossref; IEEE Xplore abstract | A |
| lambelho2020 | TU Delft research portal (abstract); details via [zoutendijk2021] | A/S |
| zoutendijk2021 | Crossref; MDPI full text (browser) | F |
| friedman2001, hastie2009, davis2006, gneiting2007, zadrozny2002, saerens2002 | Crossref metadata | M |
| chen2016 | Crossref; arXiv abstract | A |
| grinsztajn2022 | NeurIPS / ML Anthology record; abstract via index | A |
| saito2015 | PLOS ONE abstract (browser) | A |
| brier1950 | multiple bibliographic sources (DOI, volume, pages) | M |
| platt2000 | multiple bibliographic sources; book dated 1999 or 2000 depending on source | M |
| niculescumizil2005 | Crossref; abstract via ML Anthology index | A |
| guo2017 | PMLR proceedings page | A |
| vancalster2019 | Crossref; BMC Medicine abstract (browser) | A |
| kaufman2012 | Crossref; abstract via index | A |
| bergmeir2012 | Crossref; abstract via Monash/UGR records | A |
| morenotorres2012 | Crossref; abstract via index | A |
| field2007 | Crossref; abstract via ANU/IDEAS | A |
