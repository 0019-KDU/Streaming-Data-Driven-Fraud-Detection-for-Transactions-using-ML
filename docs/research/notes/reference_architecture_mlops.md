# Production Reference Architecture and MLOps for Real-Time Transaction Fraud Detection

Research date: 2026-11-25. Many sources are vendor blogs (Confluent, Tecton, Redis, AWS, Alibaba, Tacnode), each with a commercial interest. Bias is flagged where it matters. The Stripe material comes from 2015–2018 talks and the current Radar guide.

## Q1. What does a well-designed real-time fraud scoring path look like, and where does each component sit?

### Takeaway
The canonical layered design runs as follows: event ingestion (Kafka) → stateful stream processing for velocity/aggregate features (Flink, typically) → low-latency online feature store (Redis/DynamoDB/Cassandra-class KV) → synchronous scoring service (GBDT model plus rules) → decision (approve/review/block/step-up) → case management and investigator feedback → offline store/lakehouse for training. Flink is usually preferred over Spark micro-batch because it processes each event continuously rather than in trigger-interval batches.

### Cited Findings
**Layered control model (academic reference, ULB handbook)**
- The Fraud Detection Handbook (Le Borgne et al.) describes five layers. (1) Terminal checks (PIN, card status, balance, limits) must respond in "a few milliseconds". (2) Transaction-blocking rules use only data available at payment time, with no history or profile lookup. (3) Expert scoring rules run on enriched feature vectors. (4) A data-driven model (classifier) outputs a fraud probability. (5) Human investigators review alerts in a case management tool — [Fraud Detection Handbook, Ch.2 FDS](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_2_Background/FDS.html)
- The first two layers run in real time before authorization. The scoring rules and the data-driven model run in near real time and can block a card to stop further fraud. Investigators work offline — [Fraud Detection Handbook, Ch.2](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_2_Background/FDS.html)
- Investigators can check only a few alerts per day, so the model's main goal is precise alerts within an "alert budget". Too many false alarms lead investigators to ignore alerts — [Fraud Detection Handbook, Ch.2](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_2_Background/FDS.html)

**Canonical streaming stack**
- A vendor article describes the common pattern as Kafka → Flink → feature store → model serving → rules engine. Failure points under load include stale velocity counters and drift between the feature store and the rules cache — [Tacnode](https://tacnode.io/post/real-time-fraud-detection-architecture) (vendor)
- Alibaba Cloud's reference architecture uses Flink keyed state and windowing for features, Tair (a Redis-compatible store) for persistence across restarts, and Lindorm as the offline/historical store — [Alibaba Cloud](https://www.alibabacloud.com/blog/real-time-fraud-detection-pipeline-architecture-on-alibaba-cloud_603297) (vendor)
- AWS pairs MSK (Kafka) streaming ingestion with SageMaker Feature Store, which claims single-digit-millisecond online reads. AWS stresses that "up-to-the-minute" features are needed to respond to an ongoing attack — [AWS ML Blog](https://aws.amazon.com/blogs/machine-learning/use-streaming-ingestion-with-amazon-sagemaker-feature-store-and-amazon-msk-to-make-ml-backed-decisions-in-near-real-time) (vendor)
- Microsoft Fabric's fraud reference architecture has phases for ingest/process, transform/enrich, and train/score — [Microsoft Learn](https://learn.microsoft.com/en-ca/Fabric/real-time-intelligence/architectures/fraud-detection) (vendor)
- An Apache Fluss community blog shows a Flink job that consumes transactions continuously, flags fraud in real time, and writes history to Iceberg for analytics. This is an emerging 2025–26 "streaming lakehouse" pattern — [Apache Fluss blog](https://fluss.apache.org/blog/fluss_fraud_detection/)

**Flink vs Spark Structured Streaming vs micro-batch**
- Confluent says Spark's batch boundary adds a periodic delay ("near real-time … typically in the seconds range"), while Flink's continuous pipeline reaches "sub-second or millisecond" latency. Confluent also lists Spark continuous processing as experimental and says Flink is gaining ground for fraud detection — [Confluent compare](https://www.confluent.io/compare/spark-streaming-vs-flink) (vendor with a strong Flink interest)
- A practitioner estimate puts the Spark Structured Streaming trigger interval at roughly 100 ms–5 s under production load. This is anecdotal, not a benchmark — [DEV Community](https://dev.to/thejoud1997/4160-days-system-design-questions-1lnm)
- Kai Wähner reports that Spark "Real-Time Mode" arrived with Apache Spark 4.1 (stateless queries first) and reached GA on Databricks in 2026. He places instant-payment fraud detection in the "tens to hundreds of ms" latency tier — [DZone, Kafka/Flink/Spark latency SLA](https://dzone.com/articles/kafka-flink-spark-latency-sla) (not independently confirmed; check Spark release notes)
- Spark has event-time and watermark support, but Confluent notes it was added later to the micro-batch model. In Flink, event-time and watermarks are native — [Confluent compare](https://www.confluent.io/compare/spark-streaming-vs-flink)

**Stripe as an example production system**
- Stripe Radar uses "hundreds of features", many aggregated across its network, such as card country, IP, the two most frequent IPs seen with a card, and merchant embeddings. 90% of cards have been seen more than once on the network. Counters must update fast because they sit inside the API flow — [Stripe Radar guide](https://stripe.com/en-gb-us/radar/guide)

### Inferences
- A defensible reference path: synchronous authorization request → scoring API → (a) read precomputed streaming aggregates from the online KV store, (b) compute request-time features in-process, (c) run the GBDT model plus a rules engine → decision. Flink updates the KV store asynchronously from the Kafka event stream. Every event and decision is written back to Kafka/lakehouse for training, monitoring and audit.
- Kafka Streams is a reasonable lighter-weight option when the team is JVM/Kafka-centric and the topology is simple. I found no fraud-specific comparison source for it (see Gaps).

### Gaps
- No independent (non-vendor) benchmark comparing Flink, Spark RTM and Kafka Streams end-to-end latency on a fraud workload.
- Chip Huyen's "Designing ML Systems" and Google "Rules of ML" were not fetched in this session, so none of their claims are cited here.

## Q2. Why is a synchronous request/response scoring path required for payment authorization, and when is async streaming acceptable?

### Takeaway
To prevent fraud (as opposed to detecting it after the fact), the decision must be returned inside the authorization round-trip. That round-trip has hard issuer and network timeouts, so the scorer is a synchronous, in-line service with a p99 budget of tens of ms inside an end-to-end budget of about 100–200 ms. Async streaming scoring fits post-authorization actions: card blocking, alerting investigators, account-level and AML monitoring, and scoring non-blocking events.

### Cited Findings
- Visa says fraud scoring happens "within milliseconds" and that issuers use the score to approve or decline during authorization — [Visa: What is fraud scoring](https://www.visa.com/en-us/thought-leadership/risk-fraud/what-is-fraud-scoring)
- Vendor and practitioner sources put end-to-end authorization at about 100–200 ms, with the fraud-scoring step at about 10–50 ms in high-performance deployments. Redis is one source; the DEV posts are practitioner claims — [Redis blog](https://redis.io/blog/real-time-fraud-detection/); [DEV: 47ms fraud scoring](https://dev.to/amir_shachar_bc46a63dda21/real-time-fraud-scoring-latency-what-47ms-actually-means-3700)
- A system design write-up uses a p99 budget of 100 ms from receipt to decision for in-line marketplace fraud — [AlgoMaster](https://algomaster.io/learn/ml-system-design/design-fraud-detection-system)
- One source says issuers work with hard timeout windows "typically in the range of a few hundred milliseconds", and that network hops plus issuer processing use most of the budget — [Yuno blog](https://www.y.uno/en/blog/the-real-cost-of-cross-border-authorization-latency) (vendor; no network-rulebook figure found)
- The handbook distinguishes real-time pre-authorization layers (terminal and blocking rules) from near-real-time scoring that can block a card for *future* transactions. This is the classic split between synchronous and asynchronous scoring — [Fraud Detection Handbook, Ch.2](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_2_Background/FDS.html)

**Serving runtimes for tree models**
- A single third-party benchmark (one 50-tree XGBoost model on an Apple M2 Pro, in-process) reports per-sample latency of about 670 µs for Python XGBoost, about 80–150 µs for ONNX Runtime, about 10–30 µs for Treelite, and about 2 µs for AOT-compiled C (Timber). Treat it as indicative only — [Timber summary](https://news.lavx.hu/article/timber-aot-compiling-classical-ml-models-to-native-c-for-microsecond-inference)
- The Hummingbird paper (arXiv) notes that ONNX-ML is optimized for single-record, single-core inference, which fits per-transaction scoring — [arXiv 2010.04804](https://arxiv.org/pdf/2010.04804)
- Treelite's own documentation warns that for datasets with fewer than about 30 features and few missing values, Treelite may give no gain — [Treelite benchmark docs](https://treelite.readthedocs.io/en/0.31/benchmark.html)
- NVIDIA Triton's FIL backend serves XGBoost, LightGBM and Treelite checkpoints. Its dynamic batching is tuned with `max_queue_delay_microseconds` — [AWS blog: Triton for tree models](https://aws.amazon.com/blogs/machine-learning/achieve-low-latency-hosting-for-decision-tree-based-ml-models-on-nvidia-triton-inference-server-on-amazon-sagemaker); [NVIDIA dev blog](https://developer.nvidia.com/blog/?p=43509)

### Inferences
- Model inference for GBDTs takes microseconds to about 1 ms. The p99 budget is dominated by feature retrieval (network round-trips to the KV store), serialization and tail latency, not by the model. Teams therefore co-locate the feature store, batch feature lookups into one multi-get, and set a fail-open or fail-to-rules fallback when the scorer exceeds its timeout.
- Async scoring through Flink/Kafka is acceptable when the action does not need to stop the current transaction. Examples are post-auth card blocks, investigator alerting, account takeover sweeps and AML.

### Gaps
- No official Visa/Mastercard rulebook figure for the authorization timeout was found. The "few hundred ms" figure is vendor-reported.
- No independent multi-runtime p99 benchmark under network load was found.

## Q3. How do teams prevent training/serving skew for streaming aggregate features?

### Takeaway
They define features once, in a feature platform, and compute them from the same logic for both online and offline use (or backfill offline from the stream logs). Training sets are built with point-in-time-correct ("as-of") joins keyed on event timestamp, and preferably also on created/available timestamp, so that no future information leaks in. Sliding-window tiling with a defined slide interval makes the freshness that training assumes match what serving actually sees.

### Cited Findings
- Feast `get_historical_features` does point-in-time joins: for each entity row it scans *backward* from that row's timestamp up to the feature view's TTL. "TTL is not relative to the current point in time" — [Feast docs: point-in-time joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins)
- By default Feast uses created_timestamp only to deduplicate, so a value backfilled or corrected later can still leak into an earlier training row. `filter_by_created_timestamp=True` adds a `created_timestamp <= entity_timestamp` condition so each row sees only values that were *available* at the time — [Feast docs](https://docs.feast.dev/getting-started/concepts/point-in-time-joins)
- Tecton/Airbnb authors say real-time aggregation is hard partly because "the need for a separate compute path to train models using offline feature values" creates skew — [TDS: Real-Time Aggregation Features Pt 1 (Tecton/Airbnb authors)](https://towardsdatascience.com/real-time-aggregation-features-for-machine-learning-part-1-ec7337c0a504)
- In Tecton, the slide interval determines how often streaming jobs emit updated values, and so bounds freshness. Sliding windows store partial aggregations ("tiles", size = aggregation_interval). Continuous windows store every event and read them all at request time, which is freshest but costs most — [Tecton docs: aggregation windows](https://f3c685ea.tecton-docs.pages.dev/docs/defining-features/feature-views/aggregation-engine/aggregation-windows)
- Windows longer than stream retention must be backfilled from the batch source. Otherwise a new long window, for example 6 weeks, serves inaccurate values until enough history has accumulated — [Tecton/TDS Pt 2](https://towardsdatascience.com/real-time-aggregation-features-for-machine-learning-part-2-fe9fd42522c0/)
- Feature stores address consistency between offline batch-computed training features and online real-time features — [Conduktor glossary](https://www.conduktor.io/glossary/real-time-fraud-detection-with-streaming)
- Databricks markets "sub-second freshness" feature serving — [Databricks blog](https://www.databricks.com/br/blog/how-databricks-feature-store-serves-features-sub-second-freshness) (vendor)

### Inferences
- Skew-prevention practices:
  - Compute velocity and aggregate features with one definition that drives both the streaming job and the backfill.
  - Log the actual served feature vector at decision time. This "log-and-wait" approach lets the next model train on exactly what was served.
  - Use as-of joins with event time *and* availability time.
  - Test parity by replaying recent traffic offline and diffing against logged online values.
- Online stores commonly cited include Redis/Tair, DynamoDB, Cassandra and SageMaker Feature Store's online tier. Hopsworks and Featureform are alternatives, but I did not fetch their docs this session.

### Gaps
- No quantitative skew incident data from a fraud team was found.
- Hopsworks/Featureform specifics and online-store latency comparisons (Redis vs DynamoDB vs Cassandra) were not sourced.

## Q4. How do teams handle the fact that blocked transactions never get labels (and label delay generally)?

### Takeaway
There are two problems. (a) **Label delay**: chargebacks arrive weeks to months later (Visa issuers generally have 120 days, with exceptions up to 540), so training excludes a recent "immature" window (Stripe used about 60 days; the ULB handbook uses a 1-week delay block in its simulations). (b) **Selection bias**: blocked transactions have no outcome. The standard fix is a small randomized "allow" holdout with propensity scores logged, then inverse-propensity weighting (IPW) for evaluation and retraining. Investigator feedback is a further, biased source of fast labels.

### Cited Findings
- Stripe: "we need to obtain statistically sound estimates of what would have happened even to the payments we blocked" — [Stripe Radar guide](https://stripe.com/en-gb-us/radar/guide)
- Stripe's counterfactual evaluation (Michael Manapat) works as follows:
  - For transactions the model would block (score > 50), a random fraction is let through. The first version used a uniform 5%, so each allowed transaction stands for about 20 blocked ones.
  - Each transaction allowed with probability p gets weight 1/p, and precision and recall are computed from weighted counts.
  - Later versions used a *decreasing propensity curve*: sampling is concentrated near the decision boundary and very sparse at high scores (about 1 in 10,000 at score 100). This was funded from a fixed fraud-loss budget (about $10k/month in the example).
  — [InfoQ: Stripe ML models fraud](https://www.infoq.com/presentations/stripe-ml-models-fraud); [SlideShare: Counterfactual evaluation](https://de.slideshare.net/MichaelManapat/counterfactual-evaluation-of-machine-learning-models)
- Caveats from the same talk:
  - Fraudster retries break independence, so the payment session should be the unit of analysis.
  - Steep propensity curves create rare high-weight rows that can swing a 90-day metric.
  - Bootstrap the logged data for confidence intervals.
  - More exploration means narrower confidence intervals but more fraud loss.
  — [InfoQ](https://www.infoq.com/presentations/stripe-ml-models-fraud)
- Label maturity: Stripe leaves a buffer of about 60 days between the latest training or validation transaction and the present — [InfoQ](https://www.infoq.com/presentations/stripe-ml-models-fraud). This dates from the 2015–18 talks, so the current value is unknown.
- The weighted log also lets teams evaluate *arbitrarily many* alternative models and thresholds offline instead of running each as a champion/challenger A/B test — [PyData Seattle 2015](https://pydata.org/seattle2015/schedule/presentation/17/); [InfoQ](https://www.infoq.com/presentations/stripe-ml-models-fraud)
- Stripe's docs acknowledge that for allow-rules there is "no way of knowing which previously-blocked charges would, if allowed, have turned out to be fraudulent" — [Stripe docs: risk evaluation](https://docs.stripe.com/radar/risk-evaluation)
- Visa dispute window: issuers generally have 120 days from transaction processing date (or expected delivery/service date), extendable to 540 days in some cases. These figures come from third-party chargeback vendors, not the Visa rulebook — [Chargebacks911](https://chargebacks911.com/chargeback-rules/chargeback-time-limit/visa-chargeback-time-limit/); [Adyen dispute timeframes](https://docs.adyen.com/risk-management/understanding-disputes/dispute-timeframes)
- Reason code 10.4 covers card-absent fraud the cardholder did not authorize — [Chargebacks911: reason code 10.4](https://chargebacks911.com/chargeback-reason-codes/visa/10-4-card-absent-environment/)
- Academic framing: verification latency, class imbalance and concept drift are the core challenges, and only a small share of transactions is checked in time — [Dal Pozzolo et al., TNNLS 2017](https://boracchi.faculty.polimi.it/docs/2017_FraudsTNNLS.pdf)
- Dal Pozzolo et al. (IJCNN 2015) train separate classifiers on immediate investigator feedback and on delayed labels, then combine them, which improved alert precision on real data — [Dal Pozzolo et al. 2015](https://boracchi.faculty.polimi.it/docs/2015_04_Credit_Card_Fraud_Detection_DalPozzolo_Boracchi_Caelen_Alippi_Bontempi.pdf)
- Investigator feedback itself introduces sample selection bias (the "alert-feedback interaction"), because only alerted transactions get labelled quickly. One detected fraud can label several recent transactions on the same card — [Fraud Detection Handbook, Ch.2](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_2_Background/FDS.html); [TNNLS 2017](https://boracchi.faculty.polimi.it/docs/2017_FraudsTNNLS.pdf)
- Handbook validation protocol:
  - Time blocks with δtrain = δdelay = δtest = 7 days. The one-week delay block excludes immature labels, and older blocks are dropped for concept drift.
  - Prequential validation shifts train and validation windows back one block per fold (4 folds), described as "the most robust way to perform model selection".
  — [Fraud Detection Handbook, Ch.5 Validation Strategies](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_5_ModelValidationAndSelection/ValidationStrategies.html)
- A 2026 arXiv preprint frames delayed and censored labels as structural problems for card fraud label reconstruction — [arXiv 2605.29272](https://arxiv.org/html/2605.29272)

### Inferences
- Production practice:
  - Log the decision, score, propensity and served features for every transaction.
  - Keep a small randomized exploration or holdout slice, focused near the threshold and funded from an explicit loss budget.
  - Train and evaluate only on matured windows; the maturity window is set from the empirical chargeback-arrival curve, often 60–120 days for card-not-present.
  - Use fast proxy labels (investigator verdicts, customer confirmations, refunds) for monitoring, knowing they are biased.
  - Without a holdout, a model's own blocks censor its labels, and the next model inherits that feedback loop.

### Gaps
- No public 2025–26 figure for current exploration rates at major processors.
- No primary Visa/Mastercard rulebook excerpt; the chargeback figures are from secondary vendor sources.

## Q5. How are decision thresholds chosen and re-tuned in production (plus decisioning, monitoring, deployment and governance)?

### Takeaway
Thresholds are set by economics: block when the expected fraud loss exceeds the expected margin and friction cost of a false decline, which gives a break-even precision. Thresholds are often set per merchant or segment, with a middle "review" or step-up band. They are re-tuned using counterfactual (IPW) logs and guardrail metrics (block rate, false positive rate, authorization rate). New models go through shadow mode → champion/challenger → governed promotion. Monitoring combines PSI-style drift on inputs and scores with matured-label performance. Governance in the US moved from SR 11-7 to SR 26-2 in April 2026.

### Cited Findings
**Threshold by business cost**
- Stripe's worked example:
  - A $26 sale at 8% margin earns $2.08 profit.
  - A fraud costs $26 + $15 chargeback fee = $38.92, which equals about 18.71 legitimate sales.
  - Break-even precision is therefore 1/(1+18.71) ≈ 5.07%. Blocking is worthwhile when precision at the threshold exceeds this.
  — [Stripe Radar guide](https://stripe.com/en-gb-us/radar/guide)
- Raising the block threshold raises precision and lowers recall. Example: at P(fraud) > 0.7, precision is 0.66, recall 0.8 and FPR 0.4 — [Stripe Radar guide](https://stripe.com/en-gb-us/radar/guide)
- Stripe's initial threshold of 50 was chosen from validation ROC. Alternative thresholds (for example 40 or 62) can be evaluated offline from the IPW-weighted log without deploying them — [InfoQ](https://www.infoq.com/presentations/stripe-ml-models-fraud)
- Stripe keeps the share of transactions above each merchant's block threshold stable across model releases. That share is the per-merchant block-rate guardrail — [Stripe Radar guide](https://stripe.com/en-gb-us/radar/guide)

**Rules + ML hybrid and decision outcomes**
- Radar lets businesses write rules, for example block if amount > $1,000 and IP country ≠ card country. Before a rule goes live, Radar shows the historical matches that were disputed, refunded or accepted. A targeted rule "will typically increase recall with little cost to precision" when ML misses a business-specific pattern — [Stripe Radar guide](https://stripe.com/en-gb-us/radar/guide)
- Flagged payments can be sent to manual review instead of being blocked or allowed outright. This can improve precision without hurting recall, but it adds human work and depends on reviewer accuracy — [Stripe Radar guide](https://stripe.com/en-gb-us/radar/guide)
- The handbook describes expert blocking and scoring rules alongside the data-driven model. Its rules are subjective and hard to maintain, while the model cannot be edited directly by investigators — [Fraud Detection Handbook, Ch.2](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_2_Background/FDS.html)
- Some recent systems send very high-risk transactions straight to cardholders for confirmation, which works as step-up or fast feedback but must be handled carefully to protect customer trust — [Fraud Detection Handbook, Ch.2](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_2_Background/FDS.html)

**Safe deployment**
- In a Thoughtworks bank card-application fraud case:
  - Certified models run in "shadow mode" on live traffic without influencing decisions, and their scores are stored for comparison.
  - A challenger that outperforms goes through governance review before promotion to "champion".
  - Any change to rules, models or strategies requires governance review.
  — [Thoughtworks: CD for ML to tackle fraud](https://www.thoughtworks.com/en-us/insights/blog/using-cd-machine-learning-models-tackle-fraud)
- Before release, Stripe checks false positive rate, block rate and authorization rate overall and per merchant on a subset of users. It retrains monthly, which can add "as much as half a percentage point" of recall per month, and says it has tripled model release speed — [Stripe Radar guide](https://stripe.com/en-gb-us/radar/guide)

**Monitoring**
- The common PSI bands are < 0.1 stable, 0.1–0.25 moderate, and > 0.25 significant. Qlik recommends retraining at ≥ 0.25 — [Arize glossary](https://arize.com/glossary/population-stability-index-psi/); [Qlik docs](https://help.qlik.com/en-us/cloud-services/Subsystems/Hub/Content/Sense_Hub/AutoML/monitor-drift.htm)
- Critique: a Journal of Risk Model Validation paper says the 0.10 and 0.25 cut-offs have no academic support or reference to type I/II error rates, and that PSI interpretation should account for sample size and bin count — [Risk.net, Statistical properties of the PSI](https://www.risk.net/journal-of-risk-model-validation/7725371/statistical-properties-of-the-population-stability-index)
- PSI detects distribution change, not performance loss. Apply it to model *outputs* (score drift) as well as inputs — [Arize glossary](https://arize.com/glossary/population-stability-index-psi/)
- OpenTelemetry-based monitoring of fraud inference latency is described in a 2026 post — [OneUptime](https://oneuptime.com/blog/post/2026-02-06-monitor-fraud-detection-inference-latency-opentelemetry/view)

**Governance / regulation**
- On April 17, 2026, the Federal Reserve, FDIC and OCC issued revised interagency model risk guidance, SR 26-2 (OCC Bulletin 2026-13). It supersedes SR 11-7 (2011) and SR 21-8 (the BSA/AML model statement) — [Federal Reserve SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/sr2602.htm); [EY summary](https://www.ey.com/content/dam/ey-unified-site/ey-com/en-us/insights/banking-capital-markets/documents/ey-model-risk-management-guidance.pdf)
- Commentators' reading of SR 26-2:
  - It is shorter and principle-based, with materiality as the organizing principle.
  - The fixed "at least annual" review is dropped in favor of risk-based timing.
  - Requirements for development, documentation, validation, ongoing monitoring and third-party oversight remain.
  - Its principles apply to non-generative ML fraud models, while generative and agentic AI are reportedly excluded from formal scope.
  — [MathWorks blog](https://blogs.mathworks.com/finance/?p=3009); [CRA Insights](https://media.crai.com/wp-content/uploads/2026/07/09165122/FE-Insights-MRM-Guidance-SR-26-2-In-the-Era-of-AI-July2026.pdf); [Moody's](https://www.moodys.com/web/en/us/insights/banking/from-sr117-to-sr262-managing-model-risk-when-models-dont-stand-still.html); [KPMG](https://kpmg.com/us/en/articles/2026/model-risk-management-revised-interagency-guidance-reg-alert.html)
- EY lists a 2027 effective date (not verified against agency text) — [EY](https://www.ey.com/content/dam/ey-unified-site/ey-com/en-us/insights/banking-capital-markets/documents/ey-model-risk-management-guidance.pdf)
- SR 11-7 original text, for historical reference — [Federal Reserve SR 11-7](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm)

### Inferences
- Re-tuning loop:
  - Recompute the cost curve (expected loss = P(fraud) × amount + fees, against margin plus friction of a false decline or step-up) on matured, IPW-weighted data.
  - Choose a block threshold and a review/step-up band per segment, constrained by investigator capacity (the alert budget) and per-merchant block-rate guardrails.
  - Validate in shadow, then promote through champion/challenger with a holdout.
- Dollar-weighted metrics (fraud dollars caught, false-decline dollars) matter more than count-based AUC, because loss scales with amount (as in Stripe's per-sale cost example).
- Under SR 26-2 and the older SR 11-7, fraud models need inventory, documentation, independent validation and ongoing monitoring. Persisted decision logs (features, score, model version, rule hits, reason codes) are the practical evidence base.

### Gaps
- **SHAP / reason codes**: I found no primary source this session on SHAP-based reason codes for fraud analysts, or on whether ECOA/Reg B adverse-action rules apply to fraud declines. That needs a separate search.
- **PCI-DSS v4.0**: not researched this session (tokenization, PAN masking in feature stores and logs, scope reduction). Not cited.
- **Case management**: queue SLAs, staffing and tooling, beyond the handbook's "few alerts per day", were not found.
- **3DS / step-up rates** and friction cost estimates were not found.
- Chip Huyen's DMLS and Google Rules of ML were not fetched.
- Fraud-specific PSI thresholds were not found. Generic thresholds are a rule of thumb.
