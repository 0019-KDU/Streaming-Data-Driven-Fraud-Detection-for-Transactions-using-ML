# Modeling and Evaluation Pitfalls in Fraud Detection ML (with focus on IEEE-CIS)

Evidence labels: **[PR]** = peer-reviewed; **[Preprint]** = arXiv, not confirmed peer-reviewed; **[Docs]** = official documentation or vendor guidance; **[Practitioner]** = competition write-up, blog, or opinion. All sources were accessed 2026-11-25.

## Q1. What is the recommended validation protocol for fraud models (train / early-stop / calibrate / threshold / test, all chronological)?

### Takeaway
Every split should run forward in time. Leave a gap ("delay") between the training and evaluation windows so that it matches how late fraud labels arrive. Run model selection with prequential or rolling forward folds. Early stopping, calibration, threshold choice and final testing should each use their own later, disjoint slice. A random split, or reusing one slice for several of these jobs, gives scores that look better than the model really is.

### Cited Findings
- The Fraud Detection Handbook (Le Borgne et al., online book, c. 2021–2022) recommends **prequential validation** as the most robust way to choose a model. In its examples, each of 4 folds moves the training and validation windows back by one 7-day block, and the validation window is as long as the test window. The cost is that several models must be trained for each hyperparameter setting. [Practitioner/academic textbook] — [Handbook Ch.5 Validation Strategies](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_5_ModelValidationAndSelection/ValidationStrategies.html)
- The Handbook puts a **7-day delay period** between training and test because fraud labels arrive late, so the most recent transactions cannot be labelled at training time. Its layout is training week, then delay week, then test week. Hold-out validation shifts this setup back 14 days (delay plus validation). The Handbook calls repeated random hold-out "less suitable for time-ordered data." — [Handbook Ch.5](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_5_ModelValidationAndSelection/ValidationStrategies.html)
- In the Handbook's experiments, choosing tree depth by validation AUC-ROC picked depth 50, which did poorly on test. AP and Card Precision@100 were more reliable for model selection, and test AP peaked at depths of about 3–8. — [Handbook Ch.5](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_5_ModelValidationAndSelection/ValidationStrategies.html)
- Google's Rules of ML, Rule #33: "train on earlier data and test on later data" (for example, train through Jan 5 and test from Jan 6). Performance should be somewhat worse on the newer data, but not radically worse. [Docs/practitioner, Google, c. 2017+] — [Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)
- The 1st-place IEEE-CIS team (Deotte and Yakovlev, 2019) did not trust any single validation scheme. They used several time-based splits, such as "train 2 months / skip 2 / predict 2" and "train 4 / skip 1 / predict 1". Final models used GroupKFold with month as the group (6 folds). When the forward-in-time holdout disagreed with CV, they trusted the holdout. They also tracked performance separately for known, unknown and "questionable" UIDs. [Practitioner; this is a Korean repost of the original Kaggle discussion #111284] — [eda-ai-lab repost](https://eda-ai-lab.tistory.com/448); original thread (did not render for fetching): [Kaggle #111284](https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/111284)
- NVIDIA blog (McDonald and Deotte, 26 Jan 2021): the 1st-place pipeline used GroupKFold over ordered months, so that training data comes before test data in time. Its holdout trained on the first 75% of the data and tested on the last 25%. The lesson it gives: "use time-based splits, not random splits." [Practitioner] — [NVIDIA blog 2021](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution)
- scikit-learn: a calibrator must be fit on data the classifier was not trained on. A classifier scores its own training data better than new data, so a calibrator fit on training outputs pushes probabilities toward 0 and 1 too hard. When calibrating an already-fitted model with `FrozenEstimator`, the user must make sure the calibration data is disjoint from the training data. [Docs] — [sklearn calibration guide](https://scikit-learn.org/stable/modules/calibration.html)
- Dal Pozzolo et al. (IEEE TNNLS, published online 2017, journal issue 2018) say most earlier work unrealistically assumes every label arrives regularly, for example daily. In real systems, labels come back through an **alert–feedback loop** with **verification latency**. This introduces sample selection bias, and the latency is especially harmful when concept drift occurs. [PR] — [Dal Pozzolo et al. TNNLS PDF](https://boracchi.faculty.polimi.it/docs/2017_FraudsTNNLS.pdf); [arXiv 1709.08920](https://arxiv.org/pdf/1709.08920.pdf)

### Inferences
- A defensible protocol for an offline dataset such as IEEE-CIS (about 6 months of training data, ordered by TransactionDT) looks like this. Months 1–3 or 4 for training. A later slice for early stopping. A further later slice for calibration and threshold selection. The last month or two as an untouched test set, ideally with a gap of about 1 week or more before it. Selection, early stopping, calibration and threshold choice must never see the test slice.
- Choosing the threshold or the number of boosting rounds on the test set, and then reporting test metrics, is optimistic leakage. The bias is small for AUC but can be large for precision/recall at the chosen threshold.
- A per-entity split matters too. Most test cards are unseen (see Q3), so evaluating on known and unknown cards separately shows whether the model generalises.

### Gaps
- I found no peer-reviewed study that measures how large the bias is when threshold or calibration is chosen on the test set, specifically for fraud models. The argument is methodological, not quantified here.
- The original Kaggle discussion pages did not render. The 1st-place details come from a repost and from the NVIDIA blog that Deotte co-authored.

## Q2. Which metrics do practitioners and the fraud handbook literature recommend reporting, and why is AUC-ROC misleading at ~3.5% prevalence?

### Takeaway
At low prevalence, AUC-ROC is dominated by false positive rates that operations could never run at. AUC-ROC can also rank models in the opposite order from AP. The recommended set is: AP / PR curve as the main threshold-free metric; precision@k or card-precision@k tied to the daily alert budget; recall at a fixed low FPR; and, where amounts are known, dollar-weighted recall or savings. AUC-ROC should still be reported for comparability (it is the IEEE-CIS leaderboard metric), but it is not enough.

### Cited Findings
- Saito and Rehmsmeier (PLOS ONE 10(3): e0118432, 2015): on imbalanced data, the PR plot is more informative than the ROC plot. ROC's visual appeal rests on an "intuitive but wrong interpretation of specificity." The evidence comes from simulations plus a literature analysis. [PR] — [PMC4349800](https://pmc.ncbi.nlm.nih.gov/articles/PMC4349800/)
- Handbook Ch.4 makes three points:
  - A random classifier scores AUC-ROC = 0.5 whatever the class balance, while the random baseline for PR is the prevalence P/(P+N). In the Handbook's test set that baseline is 385/58,264 ≈ 0.0066.
  - Investigators can review about 1% of transactions in the Handbook's example (700 reviews over 7 days out of 58,264), so any FPR above about 0.01 is beyond what they can handle. For real-world volumes the Handbook puts the practical limit near FPR 0.001, which means about 99.9% of the ROC curve has little operational value.
  - Rankings can flip. An unlimited-depth tree and a depth-2 tree score ROC AUC 0.788 vs 0.763, but AP 0.309 vs 0.496.
  — [Handbook Ch.4 Threshold-free metrics](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_4_PerformanceMetrics/ThresholdFree.html)
- The Handbook says to plot PR curves as step functions and to use AP rather than a linearly interpolated AUC-PR. AP = Σ(Rₙ − Rₙ₋₁)·Pₙ. — [Handbook Ch.4](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_4_PerformanceMetrics/ThresholdFree.html)
- **Precision top-k (P@k):** the share of the top-k daily alerts that are fraud, with k = 100 per day, averaged over test days. The rationale is that "the number of alerts that may be checked during a given period is necessarily limited." **Card Precision top-k (CP@k):** transactions are grouped per card (max score, max label), and cards already detected are removed from later days. Example results for a depth-2 tree: mean P@100 = 0.291, and mean CP@100 = 0.286 without removal or 0.241 with removal. On day 129 the best possible P@100 was 0.55, because only 55 frauds occurred that day. The Handbook prose says CP@k ≈ 0.33, which does not match its own printed code output. — [Handbook Ch.4 Top-k](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_4_PerformanceMetrics/TopKBased.html)
- Rules of ML Rule #37 splits skew into three layers: train vs holdout, holdout vs next-day data, and next-day vs live data. For the same example, the model should "give exactly the same result" in training and serving; a difference usually means an engineering error. [Docs] — [Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)
- IEEE-CIS prevalence: 590,540 training transactions, 20,663 of them fraud (about 3.5%). — [Amazon Fraud Dataset Benchmark, Grover et al. 2022, arXiv 2208.14417](https://arxiv.org/html/2208.14417v3); [FDB GitHub](https://www.github.com/amazon-science/fraud-dataset-benchmark)

### Inferences
- At 3.5% prevalence, a model with AUC-ROC 0.93 can still have modest precision at the FPR operations can afford. Reporting AP (random baseline about 0.035), recall at FPR = 0.1% / 0.5% / 1%, and precision@alert-budget makes the operational picture clear.
- Dollar-weighted recall (the share of fraud amount caught) and net savings (fraud amount caught minus review cost × alerts minus friction cost of false declines) are cost-sensitive extensions. The Handbook pages I fetched do not define cost-based metrics (see Gaps).

### Gaps
- I did not retrieve a primary source with a formal dollar-weighted / savings metric for fraud. Bahnsen et al.'s cost-sensitive work (2014–2016) is the usual citation but was not fetched, so treat this as unverified.
- I did not fetch the Eugene Yan or Chip Huyen blogs, so I make no claims from them.

## Q3. What did the IEEE-CIS top solutions do (UID "magic" feature, adversarial validation, time-consistency checks), and what AUCs are realistic?

### Takeaway
The 1st-place team rebuilt client identity with UID = card1 + addr1 + D1n, where D1n = day − D1. They did **not** feed the UID itself to the model. Instead they built group aggregates over it, removed features with a time-consistency test, and averaged predictions per UID as post-processing. The final private LB was **0.9459**, against 0.9677 public. Single models scored about 0.932–0.941 private, and a pre-UID local validation scored 0.9363. Adversarial validation AUC close to 1 showed that the *client population* changes over time, so the task rewards generalising to unseen cards.

### Cited Findings
- **UID construction:** day = TransactionDT / 86,400, D1n = floor(day − D1), uid = card1_addr1_D1n. D1n is roughly the day the card was issued or first used. The UID was not used as a feature "to avoid overfitting." [Practitioner] — [NVIDIA blog 2021](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution); [eda-ai-lab repost](https://eda-ai-lab.tistory.com/448)
- **Why not use the UID directly:** about **68.2% of private-test clients never appear in train**, so a raw UID would overfit to known clients. Aggregates generalise to new clients:
  - mean and std over the UID of C, D (D4, D9, D10, D15) and M1–M9, plus TransactionAmt, with M columns mapped T=1 / F=0;
  - counts such as uid_P_emaildomain_ct;
  - frequency encodings;
  - TransactionAmt split into dollars and cents.

  The team gave the model UID1 plus building blocks such as D15n and let the GBDT learn how to combine them. — [eda-ai-lab repost](https://eda-ai-lab.tistory.com/448); [NVIDIA blog](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution)
- **Time-consistency check:** each feature was trained alone on the first month and validated on the last month of the training data. Features with validation AUC ≤ 0.5 were dropped. Example: id_38 scored 0.61 on train but 0.36 on validation. Dropping 19 features this way improved the model. — [eda-ai-lab repost](https://eda-ai-lab.tistory.com/448); [NVIDIA blog](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution)
- **V-column reduction:** V columns were grouped by shared NaN pattern. Within correlated groups (r > 0.75), only the column with the most unique values was kept. — [NVIDIA blog](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution)
- **Adversarial validation:** a train-vs-test classifier scored AUC near 1. The team read this as the set of clients changing over time, not as fraud patterns shifting. — [eda-ai-lab repost](https://eda-ai-lab.tistory.com/448)
- **Post-processing:** each UID's predictions were replaced by the average of that UID's known train labels and its test predictions. The gain was about +0.0005 to +0.001, and the XGB public score went from 0.9602 to 0.9618. This exploits label information from the same entity, which is not available the same way in a live system. — [eda-ai-lab repost](https://eda-ai-lab.tistory.com/448)
- **Scores:**

  | Model | Public LB | Private LB |
  |---|---|---|
  | CatBoost | 0.963915 | 0.940826 |
  | LightGBM | 0.961748 | 0.938359 |
  | XGBoost | 0.960205 | 0.932369 |
  | Equal blend | 0.966889 | 0.944795 |
  | Final submission | 0.9677 | 0.9459 (1st place) |

  The NVIDIA blog also reports Deotte's local validation: 0.9363 with 216 features before the UID, and 0.9472 after. The blog's feature counts (242 / 216 / 262) do not add up consistently. — [eda-ai-lab repost](https://eda-ai-lab.tistory.com/448); [NVIDIA blog](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution)
- The public LB is the first 20% of the test set by time, which the team confirmed by probing. This explains part of the public–private gap: the private set lies further in the future. — [eda-ai-lab repost](https://eda-ai-lab.tistory.com/448)
- The final models were an ensemble of XGBoost, CatBoost and LightGBM. The XGBoost settings included max_depth 12, learning rate 0.02, 5000 trees, subsample 0.8 and colsample 0.4. — [NVIDIA blog](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution)

### Inferences
- A realistic honest target for a strong single GBDT on IEEE-CIS with UID aggregates is about **0.93–0.95 AUC** on an out-of-time split. Without UID aggregates it is about **0.92–0.94**. A claimed AUC above about 0.96 on an out-of-time test should be checked for leakage, such as a random split, test-set threshold or early-stopping reuse, or aggregates computed over the whole dataset including future rows.
- The UID post-processing and the full-dataset aggregates (train and test combined) that Kaggle solutions commonly use are transductive and use future data. They cannot be reproduced online without a feature store that computes the same aggregates using only past data.

### Gaps
- I could not verify Konstantin Yakovlev's separate write-up, or whether full-data (train+test) aggregation was used in the final pipeline. That practice is common on Kaggle, but this source does not confirm it.

## Q4. What are the risks of SMOTE and target encoding for GBDT fraud models?

### Takeaway
For strong GBDTs, SMOTE and other rebalancing usually do not improve discrimination, and they damage probability calibration. Adjusting the decision threshold, or using class weights and then recalibrating, gets the same sensitivity gains. Target or mean encoding computed in-sample, or over future rows, leaks the label and shifts predictions. Use out-of-fold encoding that only looks at past data, or CatBoost's ordered target statistics.

### Cited Findings
- Elor and Averbuch-Elor, "To SMOTE, or not to SMOTE?" (arXiv 2201.08528, 2022): balancing helps weak classifiers but **does not improve prediction performance for strong classifiers** (XGBoost, CatBoost, LightGBM), especially once the decision threshold is optimised. When balancing is used, SMOTE-like methods were not significantly better than random oversampling. [Preprint] — [arXiv 2201.08528](https://arxiv.org/abs/2201.08528)
- van den Goorbergh, van Smeden, Timmerman and Van Calster, JAMIA 29(9):1525–1534 (2022), using logistic regression: every imbalance correction tested (random under- and oversampling, SMOTE) caused **poor calibration with strong overestimation of minority-class risk**, and none improved AUC. Shifting the threshold gave similar sensitivity and specificity without the calibration damage. [PR] — [PMC9382395](https://pmc.ncbi.nlm.nih.gov/articles/PMC9382395); [arXiv 2202.09101](https://arxiv.org/abs/2202.09101)
- A follow-up by Carriero, van Smeden et al. extends this to ML models. Models built without imbalance correction had equal or better calibration. [Preprint; seen only in search snippets] — [arXiv 2404.19494](https://arxiv.org/html/2404.19494v1)
- Dal Pozzolo, Caelen, Johnson and Bontempi (IEEE SSCI 2015): undersampling shifts the class priors and biases the posterior probabilities. It keeps the ranking but distorts calibration and accuracy. They give a closed-form correction based on the sampling rate and a Bayes-minimum-risk threshold. [PR] — [IEEE Xplore 7376606](https://ieeexplore.ieee.org/document/7376606)
- Dal Pozzolo et al. (ECML PKDD 2015): whether undersampling helps depends on sample size, classifier variance, degree of imbalance and the posterior values. [PR] — [Springer](https://rd.springer.com/chapter/10.1007/978-3-319-23528-8_13)
- de la Bourdonnaye and Daniel (arXiv 2206.13152, 2022), on a large real online credit-card payments dataset: resampling methods were "inefficient," either intractable or with no substantial metric improvement. [Preprint; abstract only] — [arXiv 2206.13152](https://arxiv.org/abs/2206.13152)
- Prokhorenkova et al., CatBoost (NeurIPS 2018): greedy target statistics leak the target because each row's encoding uses its own label, which causes a "prediction shift." Ordered target statistics, computed only from preceding rows in a permutation, fix this. The paper compares greedy, holdout, leave-one-out and ordered variants. [PR] — [arXiv 1706.09516](https://arxiv.org/abs/1706.09516)

### Inferences
- For a GBDT on IEEE-CIS, the honest default is: no SMOTE; optionally scale_pos_weight or class weights; then calibrate on a separate later slice and choose the threshold on another slice. If scale_pos_weight or undersampling is used, the raw scores are not probabilities. Either recalibrate, or apply the Dal Pozzolo correction p = βp_s / (βp_s − p_s + 1), where β is the negative sampling rate.
- If SMOTE is applied before the train/validation split, synthetic points leak into validation and inflate scores. This is a common bug. SMOTE also interpolates categorical or ID-like codes (card1, addr1) into meaningless values.

### Gaps
- I found no peer-reviewed study of SMOTE with GBDTs on IEEE-CIS specifically. The van den Goorbergh evidence is for logistic regression in clinical data. The Elor preprint covers GBDTs on general benchmarks.

## Q5. How should a model be evaluated when inference-time inputs are a small subset of training features (e.g. an API receiving ~15 raw fields while the model was trained on hundreds of V/C/D features)?

### Takeaway
If features are imputed, defaulted to NaN, or recomputed in a different way at serving time, the offline AUC no longer describes the deployed system. That is training/serving skew. The honest approach has three parts:
1. Retrain and evaluate a model using only the features that will be available at decision time, computed by the same point-in-time code used in serving.
2. Or replay the offline test set through the real serving path and measure AUC/AP there.
3. Log the features used at serving time and compare them with the offline values.

### Cited Findings
- Rules of ML #29: log the features used at serving time and reuse them for training. If that is not possible for every example, do it for a small fraction to check consistency. YouTube reportedly improved quality and simplified code this way. — [Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)
- Rules of ML #31: tables joined at training and serving time can change in between, so the same example can get different predictions. Logging features at serving time is the fix; hourly or daily snapshots only partly fix it. Rule #32: reuse code between batch training and online serving, which "eliminates a source of training-serving skew." Using different languages makes sharing nearly impossible. — [Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)
- Rules of ML #37: next-day vs live data should give *exactly the same result* for the same example. — [Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)
- The IEEE-CIS 1st-place gain came mostly from UID group aggregates and frequency encodings, 0.9363 → 0.9472 local AUC. "The secret … was feature engineering." Those aggregates (UID means of C/D/M, counts, frequency encodings) need history per entity, which an API with a few raw fields cannot compute unless it has a matching online feature store. — [NVIDIA blog](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution)
- The Amazon Fraud Dataset Benchmark cuts IEEE-CIS from 393 original features to 67, based on highly voted kernels. This is one reference point for a reduced-feature setup. — [FDB GitHub](https://www.github.com/amazon-science/fraud-dataset-benchmark); [arXiv 2208.14417](https://arxiv.org/html/2208.14417v3)
- The SCARFF architecture (Carcillo, Dal Pozzolo et al.) for streaming fraud detection uses Kafka, Spark and Cassandra to handle imbalance, non-stationarity and feedback latency. It is an example of computing streaming aggregates online. [PR, per search snippet] — [Dal Pozzolo/Boracchi pages](https://boracchi.faculty.polimi.it/docs/2017_FraudsTNNLS.pdf)

### Inferences
- If the model expects hundreds of V/C/D/id features and the API sends about 15, the missing features will be NaN or defaults. Trees then follow default-NaN branches that were rarely taken in training, so the output distribution and calibration shift. Offline AUC (about 0.93+) is then not a valid estimate of production quality. Required checks:
  - (a) Replay the out-of-time test set through the API payload path, building the same ~15-field payloads, and report AUC/AP/recall@FPR.
  - (b) Train a "decision-time-available" model on just those fields plus velocity aggregates computed with past data only, and report its out-of-time metrics alongside the full model.
  - (c) Compare score distributions (PSI) and calibration between offline and online.
- Velocity and aggregate features must be point-in-time correct. Offline, use only transactions strictly before the current one, with the same window semantics as the online store (event time vs processing time, inclusive vs exclusive of the current transaction). A Kaggle-style aggregate over train+test, or over a whole group, uses future data and cannot be reproduced online.
- The public-to-private LB drop for the winner (0.968 → 0.946), and the single-model gaps, show that even skilled teams lose about 0.02 AUC further out in time. Reduced feature availability adds more loss on top of that.

### Gaps
- **I found no published, quantified figure for how much IEEE-CIS AUC drops when only about 15 raw fields are available at inference.** It has to be measured empirically, for example by retraining on the reduced feature set with an out-of-time split. Do not cite an assumed number.
- Concept-drift magnitudes for IEEE-CIS (beyond the adversarial-validation observation and the public/private gap) were not found in a primary source. The Dal Pozzolo TNNLS paper covers drift and verification latency in general, but I did not retrieve its experimental numbers.
- I did not fetch the Chip Huyen and Eugene Yan material on real-time feature skew, so there are no citations from them.
