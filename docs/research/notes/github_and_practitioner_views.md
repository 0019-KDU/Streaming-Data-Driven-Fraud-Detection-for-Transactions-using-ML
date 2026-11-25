# Open-source real-time fraud detection reference implementations and practitioner views on fraud ML

Research date: 2026-11-25. GitHub star counts and last-push dates were pulled from the GitHub REST API (`api.github.com/repos/...`) on 2026-11-25. Reddit could not be fetched directly: reddit.com JSON was blocked, and a mirror only showed a "Preparing Space" page. Reddit content below therefore comes from search-result snippets and secondary write-ups, and is marked as such. Hacker News content comes from the HN Algolia API, which returns the original comment text.

## Q1. Which GitHub repos are widely referenced as real-time fraud detection reference implementations?

### Takeaway
No single open-source repo covers the whole production loop of streaming ingest, online features, model serving, rules, delayed labels and retraining. Each well-known repo shows one part well. Flink/Kafka demos cover stateful streaming rules, Feast/Tecton/GCP cover feature stores and training/serving parity, the Le Borgne handbook covers sound evaluation under imbalance and delay, NVIDIA/AWS cover GNN scoring, and the Deotte IEEE-CIS write-up covers offline feature engineering. Many of these repos are stale (last pushed 2021–2024).

### Cited Findings

**Fraud-Detection-Handbook (Le Borgne et al., ULB)**
- Repo `Fraud-Detection-Handbook/fraud-detection-handbook`: 760 stars, last push 2024-02-07. Description: "Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook" — [GitHub API](https://api.github.com/repos/Fraud-Detection-Handbook/fraud-detection-handbook); [repo](https://www.github.com/Fraud-Detection-Handbook/fraud-detection-handbook)
- Authors are Yann-Aël Le Borgne, Wissam Siblini, Bertrand Lebichot and Gianluca Bontempi (ULB Machine Learning Group). It is built with jupyter-book. The simulated datasets live in separate repos (simulated-data-raw, simulated-data-transformed). The first seven chapters were released around January 2022 — [repo](https://www.github.com/Fraud-Detection-Handbook/fraud-detection-handbook); [book contents](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_1_BookContent/BookContent.html)
- Content strengths:
  - Concept drift is defined as "time-dependent changes in the distributions of transactions and frauds", made worse by "delayed feedbacks".
  - It warns that "the delay in fraud reporting" requires care when splitting train and validation data.
  - Fraud is "well under 1%" of transactions.
  - Classification may need to run in "as low as tens of milliseconds".
  - AUC ROC and misclassification error are "poorly suited". The book recommends several measures, including precision top-k metrics (a separate chapter).

  — [Handbook ch.2 "ML for credit card fraud detection"](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_2_Background/MachineLearningForFraudDetection.html)
- The handbook includes a "Baseline fraud detection system" chapter, so it models the practice of starting from a baseline — [Baseline chapter](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/BaselineModeling.html)

**Apache Flink "Advanced Application Patterns" fraud demo (Alexander Fedulov / Ververica)**
- Repo `afedulov/fraud-detection-demo`: 351 stars, last push 2023-05-12, "Repository for Advanced Flink Application Patterns series" — [GitHub API](https://api.github.com/repos/afedulov/fraud-detection-demo); [repo](https://www.Github.com/afedulov/fraud-detection-demo)
- Architecture:
  - Kafka, a Flink cluster, and a React + Spring Boot web app.
  - Rules are defined in JSON (grouping fields, aggregate field and function, window length, limit and operator) and sent to Flink over a "Control" Kafka topic.
  - Transactions arrive on a "Transactions" topic, and alerts go out on an "Alerts" topic, then over WebSockets to the UI.
  - A `DynamicKeyFunction` copies each event once per rule for dynamic `keyBy` partitioning. The post notes this trades horizontal scalability for data duplication.
  - Later parts cover broadcast state for dynamic rule updates, and custom windowing with timers to limit state growth.
  - No ML is involved.

  — [Flink blog, 2020-01-15](https://flink.apache.org/2020/01/15/advanced-flink-application-patterns-vol.1-case-study-of-a-fraud-detection-system/)
- The official Flink DataStream "Fraud Detection" walkthrough (a stateful per-account detector using state and timers) is in the Flink docs [try-flink/datastream](https://ci.apache.org/projects/flink/flink-docs-master/docs/try-flink/datastream/). Azure HDInsight-on-AKS publishes a variant of it — [Microsoft Learn](https://learn.microsoft.com/en-us/azure/hdinsight-aks/flink/fraud-detection-flink-datastream-api)
- Newer streaming fraud demos:
  - Apache Fluss fraud-detection blog — [Fluss blog](https://fluss.apache.org/blog/fluss_fraud_detection/)
  - A Confluent Flink + remote model inference guide that uses the IEEE-CIS data — [msg group guide](https://www.msg.group/en/publications/real-time-fraud-detection-a-hands-on-guide-to-confluent-flink/)
  - A Redpanda + RisingWave + dbt SQL-first fraud monitor — [project page](https://alwyndsouza.github.io/projects/rp-dbt-rw-fraud-monitor/)

  I did not verify the star counts or maintenance status of these.

**Confluent**
- `confluentinc/demo-scene`: 1,568 stars, last push 2026-11-23. Its own description warns: "Not all of the examples in this repository are kept up…" — [GitHub API](https://api.github.com/repos/confluentinc/demo-scene)
- Kafka/KSQL + Flink fraud pattern article — [DZone](https://dzone.com/articles/fraud-detection-with-apache-kafka-ksql-and-apache-flink). AWS Marketplace lists Confluent Cloud real-time fraud detection material — [AWS](https://aws.amazon.com/marketplace/build-learn/data-analytics/real-time-fraud-detection-confluent-cloud/)

**Feature-store vendors**
- `feast-dev/feast-gcp-fraud-tutorial`: 17 stars, last push 2022-05-31, "Resources backing the Feast fraud tutorial on GCP". `feast-dev/feast-aws-credit-scoring-tutorial`: 76 stars, last push 2021-10-27 — [GitHub API](https://api.github.com/repos/feast-dev/feast-gcp-fraud-tutorial); [GitHub API](https://api.github.com/repos/feast-dev/feast-aws-credit-scoring-tutorial)
- `tecton-ai/tecton-sample-repo`: 29 stars, last push 2025-07-30, "Tecton's Sample Repository of Feature Definitions". It needs a Tecton account to run — [GitHub API](https://api.github.com/repos/tecton-ai/tecton-sample-repo)
- `aws-samples/amazon-sagemaker-feature-store-streaming-aggregation`: 47 stars, last push 2021-07-14. It is an end-to-end streaming feature aggregation example with SageMaker Feature Store — [GitHub API](https://api.github.com/repos/aws-samples/amazon-sagemaker-feature-store-streaming-aggregation)

**Cloud vendor end-to-end labs**
- `GoogleCloudPlatform/fraudfinder`: 252 stars, last push 2026-02-20. It is a lab series covering:
  - historical data in BigQuery plus a live Pub/Sub transaction stream
  - EDA
  - batch and streaming feature engineering into Vertex AI Feature Store
  - training (BQML), a model registry and endpoint deployment
  - real-time inference with feature lookup, and model monitoring

  It is organized as numbered notebooks 00–06 — [GitHub API](https://api.github.com/repos/GoogleCloudPlatform/fraudfinder); [README](https://github.com/GoogleCloudPlatform/fraudfinder)
- `awslabs/realtime-fraud-detection-with-gnn-on-dgl`: 225 stars. It is archived, with the last push on 2024-03-31. It is an "end-to-end blueprint architecture for real-time fraud detection (leveraging graph database Amazon Neptune)" using DGL GNNs — [GitHub API](https://api.github.com/repos/awslabs/realtime-fraud-detection-with-gnn-on-dgl)
- `awslabs/fraud-detection-using-machine-learning`: the API returned "Moved Permanently", so the repo has been renamed or transferred. I did not resolve its current location — [GitHub API](https://api.github.com/repos/awslabs/fraud-detection-using-machine-learning)

**NVIDIA**
- `NVIDIA-AI-Blueprints/financial-fraud-detection`: 77 stars, last push 2026-11-12.
  - It is a three-step workflow: data preparation, model building with a GNN-based "Financial Fraud Training container", and inference on NVIDIA Dynamo-Triton.
  - Inference returns fraud scores plus Shapley values for explainability.
  - It uses synthetic data, and needs an NGC API key and NVIDIA GPUs.

  — [GitHub API](https://api.github.com/repos/NVIDIA-AI-Blueprints/financial-fraud-detection); [README](https://github.com/NVIDIA-AI-Blueprints/financial-fraud-detection)
- `nv-morpheus/Morpheus` (Morpheus SDK): 624 stars, last push 2026-08-11 — [GitHub API](https://api.github.com/repos/nv-morpheus/Morpheus)
- For research on GNN fraud methods, `safe-graph/graph-fraud-detection-papers` is a curated list with 1,909 stars, last push 2026-11-15 — [GitHub API](https://api.github.com/repos/safe-graph/graph-fraud-detection-papers)

**Kaggle IEEE-CIS 1st place (Chris Deotte and team), per the NVIDIA write-up of 2021-01-26 by Carol McDonald and Chris Deotte**
- Validation:
  - A time-based split, because the test set is "forward in time".
  - Final models used GroupKFold with calendar months as groups.
  - Feature selection trained on the first 75% of the data and tested on the last 25%.
- The "magic" UID: `card1_addr1 + '_' + floor(day - D1)`, where `day = TransactionDT/86400`. The UID itself was not used as a feature. It was used to build group aggregates (mean, std and counts per UID). Post-processing averaged predictions per UID.
- Other features: dollars/cents split of the amount, frequency encoding, and aggregates by `card1`/`addr1` combinations. The 340 V columns were reduced by grouping on NaN pattern and dropping correlated columns (r > 0.75). Final feature count was 216.
- Models: GPU XGBoost (5,000 trees, depth 12, lr 0.02), ensembled with CatBoost and LightGBM.
- Scores: local AUC went from 0.9363 to 0.9472 with UID features. Public LB 0.9677. Private LB 0.9459, which won 1st place.

— [NVIDIA Developer Blog](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution)
- A secondary Korean summary of the 1st-place discussion says the UID was kept out of the model because 68.2% of users did not appear in the private test set. UID was used for EDA, validation and post-processing instead — [eda-ai-lab blog](https://eda-ai-lab.tistory.com/448). This is secondary and translated; I did not retrieve the original Kaggle post.

**General ML-systems references**
- `chiphuyen/machine-learning-systems-design`: 10,744 stars, last push 2023-04-15. It is a booklet with exercises, not fraud-specific — [GitHub API](https://api.github.com/repos/chiphuyen/machine-learning-systems-design)

### Inferences
- A student project that combines the parts these repos cover separately (Kafka/Flink stateful rules, a feature store, GBDT serving, delayed-label handling, monitoring) would fill a real gap. The reference repos are each partial, often tied to a vendor (GCP, Tecton, NGC), and often stale.
- Deotte's write-up shows that entity-level aggregates (card/user) and time-based validation matter more than model choice. That matches the streaming design of computing per-card velocity aggregates online.
- Star counts are modest (tens to hundreds), apart from Confluent's umbrella repo and curated paper lists. "Widely referenced" in this space means cited in docs and blogs rather than heavily starred.

### Gaps
- I did not verify the specific Confluent ksqlDB fraud demo inside demo-scene, the Morpheus fraud/GNN example subfolder, the Feast tutorial README contents, or the current location of the moved awslabs repo.
- I did not retrieve the original Kaggle discussion threads by Deotte or Konstantin Yakovlev (blocked or not searched).
- No reliable usage or adoption data was found beyond stars.

## Q2. What structure and features do strong ML-system showcase repos have?

### Takeaway
Sources on what makes a strong showcase repo are mostly blogs and course vendors with commercial bias, not hiring-manager surveys. They agree on:
- a pipeline over a notebook
- an explicit business problem and a justified metric
- a README with purpose, findings, an architecture diagram and a demo (screenshots or video)
- containerization, CI/CD badges and versioned data and models

The reference repos above show what a good layout looks like: architecture images, numbered steps and run-locally instructions.

### Cited Findings
- "A learner targeting ML engineering should pick something that involves a pipeline, not just a notebook." A portfolio project should show the builder "made choices and can explain why", unlike a tutorial — [Udacity blog, updated 2026-07-16](https://www.udacity.com/blog/10-machine-learning-projects-that-will-boost-your-portfolio/) (commercial)
- Hiring managers emphasize "the reason behind the chosen evaluation metric" — [Springboard](https://www.springboard.com/blog/data-science/machine-learning-portfolio/) (commercial)
- One source recommends coding standards (PEP8), version control for code, data and models (DVC/MLflow), CI/CD (GitHub Actions) and Docker/Kubernetes — [Merinova substack, 2024](https://merinova.substack.com/p/how-to-build-a-machine-learning-portfolio)
- Others recommend CI/CD badges, Colab notebooks, screenshots or a short demo video, and a README readable by non-technical people — [KDnuggets](https://www.kdnuggets.com/the-ultimate-guide-to-building-a-machine-learning-portfolio-that-lands-jobs); [Dataquest](https://www.dataquest.io/blog/how-to-share-data-science-portfolio/)
- A 2026 article lists Kaggle portfolio red flags, said to be drawn partly from hiring managers on r/datascience and r/MachineLearning: only competition notebooks, copied kernels, missing documentation, no deployment, no real-world data — [Learnist.org, 2026](https://www.learnist.org/kaggle-portfolio-project-red-flags-2026/). This is secondhand aggregation with no linked threads.
- Example README patterns in reference repos:
  - fraudfinder: an architecture image, then numbered notebooks 00–06 — [fraudfinder README](https://github.com/GoogleCloudPlatform/fraudfinder)
  - NVIDIA blueprint: a table of contents, hardware/software requirements, an architecture diagram and a three-step workflow — [NVIDIA README](https://github.com/NVIDIA-AI-Blueprints/financial-fraud-detection)
  - Flink demo: one set of Docker build commands that "start all required services" — [fraud-detection-demo](https://www.Github.com/afedulov/fraud-detection-demo)
- Example of honest metric reporting in a portfolio repo: the title states "leakage-safe target encoding (CV AUC 0.948, LB 0.949)", giving CV and holdout side by side — [Napstyyy repo](https://github.com/Napstyyy/kaggle-credit-card-fraud-s6e11)

### Inferences
- Working checklist derived from the above (my synthesis, not a single source):
  - README opening with the problem and the business cost of FP/FN
  - architecture diagram
  - one-command `docker compose up`
  - a pinned, reproducible results table (time-based split, PR-AUC/recall at fixed FPR, baseline row)
  - tests and a CI badge
  - ADRs or design-decision docs (for example, why GBDT, why Kafka, the threshold choice)
  - load-test latency numbers (p50/p95/p99, throughput)
  - a GIF or screenshots of the dashboard
  - a "limitations" section
- Practitioner priorities (Q3) suggest that latency/SLA numbers and precision/recall at an operating point carry more weight than a high AUC.

### Gaps
- I found no rigorous survey of hiring managers on repo features.
- I found no primary sources specifically recommending ADRs or load-test results in ML portfolios. These are listed as inference.

## Q3. What do practitioners on Reddit/HN say about fraud ML in practice versus Kaggle/academic projects?

### Takeaway
Practitioners describe fraud ML as heavily domain-driven and hybrid. The pieces they mention are:
- investigator-derived rules and decision trees
- aggregates over rich customer data
- strict latency and availability SLAs
- operating points with low precision (for example, 80% recall at 50% precision)
- label delay, and chargebacks as ground truth
- high false-positive cost felt by customers

Reddit evidence here is thin because direct access was blocked. HN gave several first-person comments, which are anecdotal.

### Cited Findings
- (Anecdotal, HN, 2024-11-14, user "actusual", who runs an ML team in fintech and is hiring): "Let me know when you've shipped a fraud model that requires four 9's, 100ms latency, with 50,000 calls an hour, 80% recall and 50% precision." The comment contrasts this with LLM-chatbot résumés — [HN comment 42141546](https://news.ycombinator.com/item?id=42141546)
- (Anecdotal, HN, 2023-03-07, user "euix", formerly in banking fraud): "improvement in model performance was directly related to its invasiveness". In practice this meant "tying together every piece of knowledge… across all databases and business channels… building massive decision trees", and "even at that level the false positive rate is very high - simply because fraud is a rare event" — [HN comment 35055534](https://news.ycombinator.com/item?id=35055534)
- (Anecdotal, HN, 2023-05-29): a former Amex exec reportedly said a Cray bought for fraud detection "was a waste of money and the cheaper statistical models worked much better" — [HN comment 36119452](https://news.ycombinator.com/item?id=36119452)
- (HN thread, 2023-10-27, 388 comments) "Tell HN: Automatic fraud detection is making my life hell" shows the customer-side cost of false positives (travelers abroad, repeated 3DS challenges). A widely replied comment lists "Wrong assumptions programmers make about fraud prevention", for example that a phone number uniquely identifies a person, or that VoIP numbers indicate fraud — [HN thread 38038713](https://news.ycombinator.com/item?id=38038713); [comment 38045174](https://news.ycombinator.com/item?id=38045174)
- (HN, 2025-06-12, 78 comments) A thread on Amsterdam's attempt to build a "fair" fraud detection model shows fairness and explainability concerns in public-sector fraud ML — [HN thread 44257862](https://news.ycombinator.com/item?id=44257862)
- (Older HN, 2014, outside the date window) Commenters say banks and issuers rarely refresh fraud models, so they miss new fraud types, which makes it "a great use-case for rules". Another warns about algorithms that "work for the wrong reasons", for example geographic patterns that are really time zones — [HN 7745342](https://news.ycombinator.com/item?id=7745342); [HN 8116047](https://news.ycombinator.com/item?id=8116047)
- (Anecdotal, Reddit r/datascience thread "What is financial fraud prevention data science", 2025-03-17, seen only via search snippets of a mirror):
  - "Fraud detection in most places mean a lot of business knowledge and building decision trees for different fraud schemes uncovered by investigators", with some anomaly detection and graph work.
  - "I have worked 4yrs as a DS in Fraud. I am typecasted as a Fraud expert"
  - "tech firms are not fans of it, but you get more respect from fintech startups"

  — [r/datascience thread via mirror](https://axebps-redlib.hf.space/r/datascience/comments/1jdotmw/what_is_financial_fraud_prevention_data_science/). The full thread was not retrievable.
- Stripe Radar, a primary vendor source:
  - It evaluates 1,000+ signals and decides in under 100 ms.
  - Ground truth comes "directly from banking partners" (disputes and chargebacks), not manual labeling.
  - Stripe notes that the lost revenue from blocking a legitimate customer can outweigh the fraud cost.

  — [Stripe primer](https://stripe.com/en-ie/radar/guide); [How we built it: Stripe Radar](https://stripe.dev/blog/how-we-built-it-stripe-radar)
- Rules vs ML (vendor sources, commercial bias): most sources recommend combining both. Rules are transparent for analysts and cover the cold start while models ramp up. One vendor says the ML ramp-up is typically 1–2 months because chargebacks must arrive first — [PayPal](https://www.paypal.com/us/brc/article/fraud-prevention-with-rules-vs-machine-learning); [Sift](https://sift.com/blog/rules-vs-machine-learning-need-win/); [Checkout.com](https://checkout.com/blog/fraud-rules-engines)
- Adversarial adaptation: "fraudsters adapt your model within weeks of deployment" — [DataInterview fraud DS guide](https://www.datainterview.com/blog/fraud-data-scientist-interview-prep) (commercial interview-prep site)
- SMOTE: one newsletter argues SMOTE "can backfire—sometimes badly in fraud work" because it creates unrealistic synthetic frauds. It notes a "growing camp" arguing "the real issue is measurement, not data composition" — [ML Pills #101](https://mlpills.substack.com/p/issue-101-smotes-limitations-in-modern). Academic papers still report gains from SMOTE combined with neural networks — [arXiv 2405.00026](https://www.arxiv.org/abs/2405.00026), so evidence is mixed. Separately, SMOTE leaks privacy information — [arXiv 2510.15083](https://arxiv.org/pdf/2510.15083)
- Accuracy is misleading on the ULB credit-card dataset: 284,807 transactions, 492 frauds (0.17%). Portfolio repos switch to PR-AUC — [Medium guide](https://medium.com/@dhanukamble001/5-data-analysis-projects-that-will-get-you-hired-f0b1a1b9c0cb); the Le Borgne handbook calls AUC ROC "poorly suited" — [Handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_2_Background/MachineLearningForFraudDetection.html)

### Inferences
- Red flags in student fraud projects, consistent across the sources above (synthesized):
  - reporting accuracy or ROC-AUC alone on imbalanced data
  - random rather than time-based splits, which leak future information and ignore label delay
  - applying SMOTE before the train/test split, or reporting SMOTE gains on resampled test data
  - no rule-based or simple-model baseline
  - Kaggle-only data without any streaming or serving aspect
  - no stated operating point (threshold, FPR, review capacity)
  - no latency numbers
- Practitioners value domain rules, entity aggregates, latency/SLA and an explicit precision/recall operating point. A project showing a hybrid rules plus ML decision engine with measured latency lines up well with these priorities.

### Gaps
- Reddit threads in r/MachineLearning, r/dataengineering, r/MLQuestions and r/cscareerquestions could not be read directly (blocked), so there are no verified quotes from them.
- I found no direct, primary-sourced practitioner quote specifically about SMOTE misuse in student projects.

## Q4. Common interview system-design questions on fraud detection and the expected answer components

### Takeaway
"Design a real-time fraud detection system" is a standard ML system-design prompt. Expected components:
- requirements (latency budget, roughly 50–100 ms)
- an offline training plane and an online serving plane sharing feature definitions
- layered scoring (blocklists and rules, then the ML model, then human review)
- a feature store with point-in-time correctness
- label definition from chargebacks and disputes, with a 30–90 day delay
- imbalance handling and cost-based thresholds
- graph or ring detection offline
- champion/challenger rollout
- drift monitoring and retraining

### Cited Findings
- Interview guides recommend splitting the design into offline training and online serving planes that share artifacts and feature definitions. They describe layered scoring: Redis blocklist/allowlist lookups in under 1 ms, then ML scoring on precomputed features in about 20–50 ms. Graph analysis for fraud rings runs "periodically (not real-time) on historical data" — [AlgoMaster](https://algomaster.io/learn/ml-system-design/design-fraud-detection-system); [spawn08 system-design notes](https://spawn08.github.io/system-design-interview/ml_system_design/fraud_detection); [techinterview.org](https://techinterview.org/system-design-fraud-detection/) (commercial or community interview prep)
- Feature store: computes features "using the same logic for both training and serving". Point-in-time correct historical features are "crucial to avoid leakage". One source cautions that a feature store does not by itself remove all training/serving skew — [same interview sources](https://algomaster.io/learn/ml-system-design/design-fraud-detection-system); [educative Grokking](https://www.educative.io/courses/grokking-the-machine-learning-interview/fraud-detection-system-architecture)
- Labels: "chargebacks and disputes provide ground truth, with 30-90 day delay". A common interviewer prompt is how to define positive and negative labels and handle label delay. A human review queue "doubles as a label-generation pipeline" — [interview sources above](https://www.techinterview.org/post/3233463088/system-design-fraud-detection-system/); [prachub](https://prachub.com/interview-questions/design-a-fraud-detection-system)
- Thresholds depend on "the business cost of each type of error". Champion/challenger rollout routes about 10% of traffic to the new model and promotes it if FP and FN rates improve. Retraining cadence is "an assumption to test, not a fixed rule" — [interview sources above](https://algomaster.io/learn/ml-system-design/design-fraud-detection-system)
- Latency budgets vary across sources (under 100 ms vs 50 ms). Stripe states under 100 ms for Radar — [Stripe primer](https://stripe.com/en-ie/radar/guide); ByteByteGo case study "How Stripe Detects Fraudulent Transactions Within 100 ms" — [ByteByteGo](https://blog.bytebytego.com/p/how-stripe-detects-fraudulent-transactions) (commercial)
- Course resources include educative "Grokking the ML Interview" (fraud detection architecture lesson) and DesignGurus "Grokking the AI System Design Interview" — [educative](https://www.educative.io/courses/grokking-the-machine-learning-interview/fraud-detection-system-architecture); [DesignGurus](https://www.designgurus.io/course-play/grokking-the-ai-system-design-interview/doc/6a5da946dc24e6f1282da498) (commercial; paywalled content not read)

### Inferences
- A project can map onto this answer template explicitly with a README section titled something like "How this maps to the standard fraud system-design answer". That section would cover the rules layer, the feature store or online aggregates, model serving latency, the label-delay strategy, thresholding by cost, the rollout strategy and monitoring. It would make the repo useful in interviews directly.

### Gaps
- I did not read the Aminian & Xu "Machine Learning System Design Interview" book chapter list (it may not have a dedicated fraud chapter; unverified) or the paywalled educative/ByteByteGo content.
- Specific numbers such as 30–90 day label delay and 20–50 ms scoring come from interview-prep sites, not primary industry data. Stripe's under-100 ms figure is the only primary latency figure found.
