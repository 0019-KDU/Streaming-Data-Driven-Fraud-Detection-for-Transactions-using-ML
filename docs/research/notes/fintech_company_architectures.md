# Production Real-Time Fraud Detection Architectures at Payments / Fintech Companies

Scope: Stripe (Radar), PayPal, Adyen (RevenueProtect/Protect), Square/Block, Revolut (Sherlock), Nubank, Monzo, Capital One, Coinbase, Wise, Klarna. Researched 2026-11-25. Publication years are given for each source. Some primary pages (Revolut blog/newsroom, PayPal Medium, Coinbase blog, Couchbase) returned HTTP 403 to the fetcher. For those, the claims come from search snippets of the primary page or from secondary or vendor summaries, and they are marked that way.

## Q1. What does each company's real-time scoring path look like end-to-end, and what latency do they target?

### Takeaway
Published latency budgets fall roughly between 50 and 350 ms per decision for the whole decision path, and some components run in single-digit ms. Stripe gives <100 ms per Radar decision (2023). Revolut Sherlock gives <50 ms (vendor-reported). Coinbase gives a p99 of about 80 ms for model scoring (2020) and 150–250 ms for streaming feature freshness (2025-26, vendor). Nubank's complex multi-step flows take about 350 ms (2025), and Capital One reports <10 ms for Flink-based scoring (secondhand). The common pattern is an orchestrator that loads features, runs rules and models (often in parallel), and maps the results to an action. The features come from three tiers: on-request, streaming/near-real-time cached aggregates, and batch.

### Cited Findings

**Stripe Radar (primary, 2023 blog; 2021 guide)**
- Radar assesses "more than 1,000 characteristics" per transaction and decides in "less than 100 milliseconds". Good transactions go through, and risky ones are blocked or sent to additional security checks (step-up). — [Stripe, "How we built it: Stripe Radar" (2023-03-29)](https://stripe.dev/blog/how-we-built-it-stripe-radar)
- Base rate: fraud is about 1 in 1,000 payments. Radar "incorrectly blocks just 0.1%" of legitimate payments. — [Stripe 2023](https://stripe.dev/blog/how-we-built-it-stripe-radar)
- Model history: logistic regression first, then DNNs. In mid-2022 Stripe moved from a "Wide & Deep" ensemble (XGBoost + DNN) to a DNN-only multi-branch architecture inspired by ResNeXt. Dropping XGBoost on its own would have cost 1.5% recall, so Stripe built a DNN component to take over XGBoost's "memorization" role. Training time fell by more than 85%, to under two hours, so experiments went from overnight runs to several per day. Scaling the training data 10x gave significant gains, and a 100x experiment was in progress. Planned next steps were transfer learning, embeddings and multi-task learning. — [Stripe 2023](https://stripe.dev/blog/how-we-built-it-stripe-radar)
- Example features: amount in USD, card issuing country, number of countries the card was used in over the past 24h (a velocity feature), the card's two most frequent IPs, device-time minus UTC offset, count of countries with successful auths, and learned merchant embeddings. There are "hundreds" of features, most of them aggregated across the Stripe network. — [Stripe ML fraud primer (last updated 2021-12-15)](https://stripe.com/guides/primer-on-machine-learning-for-fraud-protection)
- Feature ideas come from reviews of past attacks, investigation reports, cross-network correlation searches and weekly reviews of dark-web fraud trends. — [Stripe 2023](https://stripe.dev/blog/how-we-built-it-stripe-radar)
- Explainability: "Risk insights" (launched 2020) shows which features contributed to a decline. It also shows purchase and shipping maps and uses Elasticsearch to look up related transactions. — [Stripe 2023](https://stripe.dev/blog/how-we-built-it-stripe-radar)

**Nubank (primary, 2025)**
- Each event, for example a PIX transfer, enters the "Defense Platform" with its context. A "Flow Orchestrator" routes it through rules and models. The orchestrator was refactored from layer-by-layer execution into a DAG using Nubank's open-source Nodely library. For PIX, more than 40 processes (rules + models) can run in parallel. — [Nubank Building, "Scaling fraud defense" (2025-07-23)](https://building.nu.com/scaling-fraud-defense-how-nubank-evolved-its-risk-analysis-platform/)
- Stack: Clojure services, Datomic on DynamoDB, and Kafka, with ML models written in Python. There are 20 full-replica "shards" in Brazil alone. Volume is about 450M events/day and about 5M internal requests/min, with >99.98% availability. More than 100 TB of logs per day go into ETL. — [Nubank 2025](https://building.nu.com/scaling-fraud-defense-how-nubank-evolved-its-risk-analysis-platform/)
- Latency: complex flows fell from 550 ms to about 350 ms after the DAG refactor. The post states no explicit SLO. — [Nubank 2025](https://building.nu.com/scaling-fraud-defense-how-nubank-evolved-its-risk-analysis-platform/)
- Before centralisation, each product team embedded its own fraud logic, which led to duplicated and inconsistent policies. — [Nubank 2025](https://building.nu.com/scaling-fraud-defense-how-nubank-evolved-its-risk-analysis-platform/); [ZenML summary (secondary)](https://www.zenml.io/llmops-database/ai-powered-fraud-detection-platform-scaling-to-450m-daily-events)
- Streaming: the Flink-based "Avalanche" stack. An "Autopilot" system uses Avalanche analytics to adjust anti-fraud measures dynamically instead of relying on static thresholds (from a search snippet of the primary post). — [Nubank Building, Avalanche stack](https://building.nubank.com/avalanche-stack-and-real-time-streaming-applications-at-nu/)
- Graph: TigerGraph reports Nubank handling 50M+ daily transactions with an 80 ms response-time requirement. This is a vendor claim. — [TigerGraph case study (vendor)](https://www.tigergraph.com/nubank-reduces-fraud-losses/)

**Monzo (primary, 2022 and 2025)**
- The reactive fraud platform has four steps: (1) choose which controls to run, for example a model only for bank transfers; (2) load features; (3) run controls; (4) apply actions. The components are: the Engine (a Go microservice with a Controls Repository and Executor); the Feature Loader (a microservice that computes features as a DAG); and the Action Applier (a stateful service that looks up actions already taken on the user or payment). — [Monzo, "Build a reactive fraud prevention platform" (2025-07-30)](https://monzo.com/blog/build-a-reactive-fraud-prevention-platform)
- Feature tiers: "Just in Time" features are computed from the payment context (e.g. the payment reference). "Near Real Time" features are precomputed asynchronously and cached (e.g. today's total spend, a velocity aggregate). "Batch" features are refreshed periodically with SQL (e.g. spend over the past year). The Feature Loader returns partial results with per-feature errors. Engineers can set a timeout, after which unresolved features are skipped (graceful degradation). — [Monzo 2025](https://monzo.com/blog/build-a-reactive-fraud-prevention-platform)
- The post gives no numeric latency budget. It only says "minimal latency on the payment hotpath" at "millions of transactions per day". Fraud runs at about 1 in 10,000 transactions. — [Monzo 2025](https://monzo.com/blog/build-a-reactive-fraud-prevention-platform)
- Older stack: "all of our fraud classifiers need to run every time a transaction is initiated". Production is mostly Go plus Cassandra, and Python services are kept thin (load a model, predict). Operational features are computed in real time from NSQ event streams or RPC. Analytics features are computed in BigQuery on a schedule and copied into Cassandra for online lookup (a custom feature store). — [Monzo, "Monzo's machine learning stack" (2022-04-26)](https://monzo.com/blog/2022/04/26/monzos-machine-learning-stack)
- A PyData London 2025 talk covers multi-task deep learning for Monzo's real-time fraud detection. — [PyData London 2025](https://cfp.pydata.org/london2025/talk/EJWBPU/); Monzo also published on representation learning for fraud — [Monzo blog](https://monzo.com/blog/representation-learning-for-enhanced-fraud-detection)

**Coinbase (primary blog via snippet; AWS talk 2020; Databricks vendor story)**
- 2020 design: Flink on EMR computed streaming features into DynamoDB. The p99 end-to-end SLA was "as low as 80ms" because models block critical user paths. — [AWS re:Invent 2020 AIM320, Coinbase talk (PDF)](https://d1.awsstatic.com/events/reinvent/2020/Detecting_fraud_faster_with_Amazon_SageMaker_featuring_Coinbase_AIM320.pdf)
- Sequence-feature framework: Tecton feature views backed by Spark Structured Streaming, with DynamoDB as the online store. Targets are 1–2 s freshness and p99 online read <100 ms. One demanding job runs <500 ms end-to-end at about 2,000 events/s. The features feed fraud and recommender models. (Primary page returned 403; figures are from search snippets of it.) — [Coinbase blog, "How Coinbase Builds Sequence Features for ML"](https://www.coinbase.com/blog/how-coinbase-builds-sequence-features-for-machine-learning)
- Later: the critical risk models moved to Spark Real-Time Mode. Feature latency is 150 ms for stateless and 250 ms for stateful aggregations, with 250+ features on one engine and online/offline consistency "up to 98%". This is a vendor claim. — [Databricks customer story (vendor)](https://www.databricks.com/customers/coinbase/lakeflow)

**Revolut Sherlock (2018 launch; vendor/press figures)**
- Sherlock scores card transactions in under 50 ms. It runs on Google Cloud with Couchbase as the in-memory store (vendor-reported, from snippets). — [Couchbase Revolut customer page (vendor)](https://www.couchbase.com/customers/revolut/); [Couchbase case study PDF](https://info.couchbase.com/rs/302-GJY-034/images/Revolut_Case_Study.pdf)
- Revolut's 2018 launch press release reports a "fourfold reduction in card fraud". — [Revolut newsroom (2018)](https://www.revolut.com/news/revolut_unveils_new_fleet_of_machine_learning_technology_that_has_seen_a_fourfold_reduction_in_card_fraud_and_had_offers_from_banks_/)
- Recent primary post on scaling a foundation model for structured data to real-time production. The fetch was blocked (403). The snippet says online scoring in transaction-time paths is "an entirely different engineering discipline" from batch scoring. — [Revolut blog, "From research to real time"](https://www.revolut.com/blog/post/fromresearchtorealtime/)
- Unverified: a secondary blog claims Sherlock uses CatBoost. No primary source confirms this. — [advisori.de (secondary)](https://www.advisori.de/blog/revolut-s-ai-strategy-roadmap-key-takeways-learnings)

**PayPal (secondary; primary Medium post blocked)**
- PayPal built a home-grown real-time graph database and graph analysis platform against organised and repeat fraud (the primary post was not readable). — [PayPal Tech Medium](https://medium.com/paypal-tech/how-paypal-uses-real-time-graph-database-and-graph-analysis-to-fight-fraud-96a2b918619a)
- Per Aerospike, PayPal needed multi-hop graph queries returned "within a few milliseconds". The results feed AI models alongside other features. This is a vendor claim. — [Aerospike blog (vendor)](https://aerospike.com/blog/where-fraud-detection-fits-on-graph-databases/)
- ATO use case: accounts link to assets (devices, IPs, etc.) as edges, and shared records flag risky users. — [PayPal Developer blog, graph usage for ATO (2023)](https://developer.paypal.com/community/blog/graph-usage-in-combating-ato-fraud-risk/)
- The fraud system monitors "thousands of signals" and scores using 500+ data points. — [PayPal Developer blog, fraud risk solutions (2025)](https://developer.paypal.com/community/blog/paypal-fraud-risk-management-solutions/)

**Capital One (secondary only)**
- Capital One moved from a microservice-based fraud detection system to Apache Flink stream processing plus ML. Real-time scoring is reported at <10 ms (secondhand account of a talk). — [DBTA, "Building a Real Time Fraud Engine for Zelle Payments"](https://www.dbta.com/Editorial/News-Flashes/Building-a-Real-Time-Fraud-Engine-for-Zelle-Payments-125247.aspx)
- Capital One evaluated Apex, Flink, Storm and Spark Streaming. A 2019 OSCON talk covered its "real-time machine learning decision platform". — [O'Reilly OSCON 2019 session](https://conferences.oreilly.com/oscon/oscon-or-2019/public/schedule/detail/79251); Kafka-based real-time risk — [Confluent talk](https://www.confluent.io/online-talk/capital-one-delivers-customer-risk-insights-in-real-time-with-stream-processing)

**Adyen (product docs only)**
- The "Machine learning: fraud risk" rule (RevenueProtect premium) predicts the probability that a payment ends in a fraudulent chargeback. A separate "bot attack risk" ML rule exists. Merchants set a block threshold on the risk level. — [Adyen docs, ML rules](https://docs.adyen.com/risk-management/configure-your-risk-profile/machine-learning-rules); [Adyen docs, "Machine learning demystified"](https://docs.adyen.com/risk-management/next-phase)
- Adyen is transitioning RevenueProtect to a new engine called "Protect". — [Adyen help](https://help.adyen.com/guides/revenueprotect-optimization-guide/transitioning-to-adyens-new-risk)

**Square/Block (blog + job postings)**
- By default, ML models trained on Square's own payment data flag payments with high or moderate risk. Sellers can add their own rules via Risk Manager. — [Block, "Empowering sellers to fight fraud" (2024-07-23)](https://block.xyz/inside/empowering-sellers-to-fight-fraud-on-their-terms)
- A Chargeback Risk ML team covers Cash App and Square. Postings mention XGBoost/LightGBM, TensorFlow/PyTorch, online/offline consistency, drift detection, rollback, graph and sequence models, and entity resolution. — [Block careers posting](https://block.xyz/careers/jobs/4680661008); [BuiltIn posting](https://www.builtinchicago.org/job/staff-applied-machine-learning-engineer-fraud-abuse/9618203)

**Wise (primary, 2026; talk 2022)**
- Wise is building a Graph Intelligence platform that computes PageRank, community detection, label propagation and composite edge scores. These are served as graph-native features into a Feature Store. Wise is also exploring GNNs (GCN, temporal GNN, GraphSAGE), sequence models, and LLMs for red-flag detection and investigation summaries. Training covers "billions of training rows across hundreds of countries". The post describes work in progress and gives no latency figures. — [Wise, "Keeping millions safe: Scaling Fincrime ML at Wise" (2026-09-22)](https://wise.jobs/blog/2026-9/keeping-millions-safe-scaling-fincrime-machine-learning-at)

**Klarna**
- Public pages only say "real-time fraud monitoring, powered by machine learning" with 150+ fraud specialists. They give no engineering detail. — [Klarna security page](https://www.klarna.com/uk/security/)

### Inferences
- The orchestrator–feature loader–controls–action pattern recurs independently at Nubank (Flow Orchestrator + DAG) and Monzo (Engine + DAG Feature Loader + Action Applier). Two other recurring choices are DAG-parallel feature fetching and partial-failure tolerance (timeouts, skipping features).
- A three-tier feature freshness scheme (request-time / streaming-cached velocity aggregates / batch historical) appears to be the norm (Monzo explicitly, and Coinbase and Monzo 2022 implicitly). Online/offline consistency is the cited pain point (Coinbase, Block).
- Stripe is the only company with a detailed public write-up of moving GBDT to DNN. Elsewhere GBDT (XGBoost/LightGBM) still shows up in tooling lists, and graph and sequence/foundation models are the 2024–2026 frontier (Wise, Revolut, Monzo, PayPal).

### Gaps
- I found no primary engineering write-ups with architecture detail for Adyen, Klarna, Capital One (only secondhand talk summaries) or Square/Block's risk serving stack.
- PayPal Medium, Revolut blog and newsroom, Coinbase blog and Couchbase pages were blocked (403). Their figures rest on search snippets or vendor pages.
- Monzo publishes no numeric latency budget, and Nubank publishes no explicit SLO.

## Q2. How do they combine rules engines with ML models, and who owns rules (analysts vs engineers)?

### Takeaway
Every company with a public write-up runs rules and ML side by side. ML supplies a risk score, and rules either override it (merchant allow/block lists) or short-circuit expensive models. Ownership differs. Stripe, Adyen and Square expose rules to merchants. Nubank says rules currently need engineering work and plans declarative configs so analysts can write them. Monzo writes controls as Starlark pure functions.

### Cited Findings
- Stripe: users can create custom allow or block rules alongside the ML score (Radar for Fraud Teams). One example rule blocks transactions over $1,000 when the IP country doesn't match the card country. Before a rule goes live, Radar shows its historical dispute, refund and acceptance stats, so rules are evaluated like models. Manual review is described as adding precision or recall at the cost of human effort. — [Stripe primer (2021)](https://stripe.com/guides/primer-on-machine-learning-for-fraud-protection); [Stripe 2023](https://stripe.dev/blog/how-we-built-it-stripe-radar)
- Nubank: detection uses hand-written rules or ML models (which are slower). "If a rule confidently flags a transaction as high-risk, the model is skipped." Writing rules "currently requires engineering work". Planned: "a declarative, configuration-based model so fraud analysts and non-engineering roles can contribute directly." — [Nubank 2025](https://building.nu.com/scaling-fraud-defense-how-nubank-evolved-its-risk-analysis-platform/)
- Monzo: there are three control types. Detectors (typically ML) output a fraud type and confidence but no action. Action Controls advocate a specific intervention. An Action Selection Control aggregates the requested actions into a final decision. Controls are Starlark (Python-dialect) pure functions, and every control's inputs and outputs are logged to BigQuery. The post doesn't say whether analysts or engineers write them. — [Monzo 2025](https://monzo.com/blog/build-a-reactive-fraud-prevention-platform)
- Adyen: the ML fraud-risk rule sits inside the merchant's risk profile with other rules. Adyen recommends "custom rules to complement your risk profile" because the ML rule won't catch every case. — [Adyen ML rules docs](https://docs.adyen.com/risk-management/configure-your-risk-profile/machine-learning-rules)
- Square: ML flags high or moderate risk by default, and sellers write their own rules in Risk Manager. — [Block 2024](https://block.xyz/inside/empowering-sellers-to-fight-fraud-on-their-terms)
- Industry framing: a rule-based first layer, then a sub-second ML layer. — [IEEE Computer Society (secondary)](https://www.computer.org/publications/tech-news/trends/machine-learning-fraud-detection)

**Decision outcomes / human review**
- Stripe: allow, block, or "additional security checks" (3DS-style step-up), plus manual review. — [Stripe 2023](https://stripe.dev/blog/how-we-built-it-stripe-radar)
- Nubank: real-time actions are block or in-app warning. The asynchronous action is opening an internal investigation case. — [Nubank 2025](https://building.nu.com/scaling-fraud-defense-how-nubank-evolved-its-risk-analysis-platform/)
- Monzo: warning screen, fraud investigation, or stopping the payment and notifying the user ("many" actions). — [Monzo 2025](https://monzo.com/blog/build-a-reactive-fraud-prevention-platform)
- Revolut Sherlock: declines the transaction and asks the user in-app to confirm (press coverage, from snippets). — [Finextra (2018)](https://finextra.com/pressarticle/76339/revolut-unveils-machine-learning-tools-to-fight-card-fraud)
- Capital One: alerts customers in real time and locks the card if necessary. — [Capital One Tech](https://www.capitalone.com/tech/machine-learning/using-machine-learning-to-re-think-the-customer-experience-paradigm/)
- Wise: investigation outcomes feed back as governed labels, and active learning is planned to route cases to human review. — [Wise 2026](https://wise.jobs/blog/2026-9/keeping-millions-safe-scaling-fincrime-machine-learning-at)

### Inferences
- A separation of "detector score" from "action policy" (Monzo explicitly, Stripe through thresholds plus rules) lets risk and ops teams change interventions without retraining models.
- Moving rule authoring from engineers to analysts through declarative DSLs or configs is an active direction (Nubank's plan, Monzo's Starlark).

### Gaps
- No source gave a rule-vs-ML decision split (the share of blocks driven by rules versus models).
- No company described its case-management tooling in detail.

## Q3. How do they handle delayed and noisy labels?

### Takeaway
Public detail is thin. Stripe gets labels automatically from disputes in its own payment flow, and it fixes selection bias (you can't observe the outcomes of blocked payments) by randomly letting a small share of would-be-blocked payments through and using inverse-propensity weights. Wise is building a versioned label platform. Revolut and Wise's 2022 talk describe nightly or daily retraining on missed fraud and false declines. No source gave exact chargeback label-delay windows.

### Cited Findings
- Stripe labels: fraud means the cardholder didn't authorise the charge, which usually surfaces as a dispute. Radar "receives 'ground truth' information directly from the usual Stripe payment flow". — [Stripe primer (2021)](https://stripe.com/guides/primer-on-machine-learning-for-fraud-protection)
- Stripe selection bias: production outcomes are visible only for payments that weren't blocked, so Stripe needs "statistically sound estimates of what would have happened even to the payments we blocked". — [Stripe primer](https://stripe.com/guides/primer-on-machine-learning-for-fraud-protection)
- Stripe counterfactual method (talks, 2015–2018): of payments scoring above the block threshold, a small fraction is allowed uniformly at random (example p = 0.05). Evaluation and training use only the allowed samples, each weighted 1/P(allowed), so reversed charges get weight 20. The same data supports threshold comparison and unbiased precision and recall estimates. — [Michael Manapat, PyData Seattle 2015](https://pydata.org/seattle2015/schedule/presentation/17/); [InfoQ/QCon.ai talk](https://www.infoq.com/presentations/stripe-ml-models-fraud); [Slides](https://de.slideshare.net/MichaelManapat/counterfactual-evaluation-of-machine-learning-models)
- A PyData LA 2018 Stripe talk compares two counterfactual approaches and says one proved more effective. The snippet doesn't say which. — [PyData LA 2018](https://pydata.org/la2018/schedule/presentation/28)
- Wise: a "versioned, auditable ground-truth" Label Platform with label quality monitoring, confidence scoring, programmatic labelling and label propagation. Investigation outcomes go in as governed labels. Planned work: label drift detection, and per-typology precision degradation alerts. — [Wise 2026](https://wise.jobs/blog/2026-9/keeping-millions-safe-scaling-fincrime-machine-learning-at)
- Wise (2022 talk by lead data scientist Luca Traverso): feature creation and labelling, closed-loop automatic retraining (at least daily), governance and interpretability. — [Traverso slides, Univ. of Tartu 2022 (PDF)](https://cs.ut.ee/sites/default/files/2022-02/Luca%20Traverso_Building%20robust%20and%20defensible%20ML%20systems%20in%20Fincrime.pdf)
- Revolut: models reportedly retrain nightly on missed fraud and wrongly declined transactions. About 30% of fraud predictions are correct (precision) and 96% of fraud is caught (recall). These are press figures. — [TechHQ (secondary)](https://techhq.com/?p=192761)
- Adyen: the training target is explicitly "fraudulent chargeback". — [Adyen docs](https://docs.adyen.com/risk-management/configure-your-risk-profile/machine-learning-rules)
- Wise uses LLMs to mine historical case narratives into structured training data. — [Wise 2026](https://wise.jobs/blog/2026-9/keeping-millions-safe-scaling-fincrime-machine-learning-at)

### Inferences
- Random-reversal holdouts with inverse-propensity weighting are the best-documented public answer to label censoring from blocked transactions. A FYP system that blocks transactions would need something similar to evaluate itself fairly.
- Fast feedback signals (in-app user confirmations at Revolut and Monzo, investigation outcomes at Nubank and Wise) probably partly stand in for slow chargeback labels. No source quantified this.

### Gaps
- No primary source gave a chargeback label maturity window (e.g. 60/90/120 days) or described how training data is cut off to allow for label maturation.
- No source described explicit noisy-label modelling (e.g. friendly fraud versus true fraud separation) in detail.

## Q4. How do they deploy new models safely?

### Takeaway
Shadow mode is the most commonly named mechanism (Nubank: all defenses; Monzo: used extensively; Wise: in its MLOps pipeline). Stripe adds score-distribution matching, per-merchant guardrails on a subset of users, and counterfactual evaluation. Monzo also backtests pure-function controls on historical data and rate-limits actions as circuit breakers.

### Cited Findings
- Nubank: "All defenses go through a shadow phase first". New rules and models run on real inputs alongside production without affecting users. — [Nubank 2025](https://building.nu.com/scaling-fraud-defense-how-nubank-evolved-its-risk-analysis-platform/)
- Monzo: shadow-mode deployments are used extensively. A model registry lets Python services hot-reload models once they're enabled. Monitoring uses Grafana (system), dbt-Slack alerts and data validation as ingestion preconditions (features), and Looker (precision, recall, fraud rates). — [Monzo 2022](https://monzo.com/blog/2022/04/26/monzos-machine-learning-stack)
- Monzo: controls are pure functions, so they can be backtested on historical data. Per-action rate limits block further actions and alert an engineer once exceeded. Controls can be modularly added or removed. — [Monzo 2025](https://monzo.com/blog/build-a-reactive-fraud-prevention-platform)
- Stripe: before release, Stripe compares the old and new score distributions so the share of transactions above each merchant's threshold stays stable. Release guardrails track false positive rate, block rate and authorization rate, in aggregate and per merchant, on a subset of users. Dashboards track candidate models. — [Stripe primer (2021)](https://stripe.com/guides/primer-on-machine-learning-for-fraud-protection)
- Stripe retraining: retraining on recent data can lift recall "as much as half a percentage point each month". Release speed tripled, and the ML gain is ">20% year over year". — [Stripe primer](https://stripe.com/guides/primer-on-machine-learning-for-fraud-protection)
- Wise: the MLOps pipeline goes from experiment to shadow-mode scoring with per-typology evaluation, and new specialised models deploy through a config-driven pipeline. — [Wise 2026](https://wise.jobs/blog/2026-9/keeping-millions-safe-scaling-fincrime-machine-learning-at)
- Block: postings name monitoring, drift detection and rollback as engineering responsibilities. — [BuiltIn/Block posting](https://www.builtinchicago.org/job/staff-applied-machine-learning-engineer-fraud-abuse/9618203)

### Inferences
- Score-distribution calibration (Stripe) matters because merchants and operations teams set fixed thresholds. Swapping in a model without recalibrating shifts block rates even when AUC improves.
- No source used the term "champion/challenger" explicitly. Shadow mode plus guarded partial rollout is the practical equivalent.

### Gaps
- No public specifics on rollout percentages, shadow-mode duration, or promotion criteria.
- Retraining cadence: only Revolut (nightly, secondary), Wise (at least daily, 2022 talk) and Stripe (roughly monthly benefit implied) are covered. Nubank, Monzo and Coinbase give no cadence.

## Q5. What business metrics do they optimize?

### Takeaway
The published metrics are precision, recall and false-positive rate, block rate, authorization or approval rate, and fraud loss rates. Stripe makes cost-weighted thresholds explicit with a break-even precision calculation. Nubank plans to show the operational cost of each defense.

### Cited Findings
- Stripe cost-weighted threshold example: a $26 sale at 8% margin yields $2.08 profit. With a $15 chargeback fee, one fraud costs about 18.71 good sales, so break-even precision is 1/(1+18.71) = 5.07%. Raising the threshold raises precision and lowers recall, and the guide treats where to set it as a business decision. Radar balances merchant margins against stable block rates. — [Stripe primer (2021)](https://stripe.com/guides/primer-on-machine-learning-for-fraud-protection)
- Stripe evaluation: PR curves, ROC/AUC, score bimodality (with a caution that bimodality alone doesn't prove quality), and per-merchant FPR, block rate and auth rate. Reported results: 0.1% false-block rate, and a 1.5% recall drop that removing XGBoost alone would have caused. — [Stripe primer](https://stripe.com/guides/primer-on-machine-learning-for-fraud-protection); [Stripe 2023](https://stripe.dev/blog/how-we-built-it-stripe-radar)
- Monzo: tracks how much fraud the controls catch or miss, and how many actions are raised and how well they stop fraud (BigQuery logging). Looker dashboards report precision and recall alongside company fraud rates. — [Monzo 2025](https://monzo.com/blog/build-a-reactive-fraud-prevention-platform); [Monzo 2022](https://monzo.com/blog/2022/04/26/monzos-machine-learning-stack)
- Nubank: plans to show product teams the operational cost of each fraud defense. — [Nubank 2025](https://building.nu.com/scaling-fraud-defense-how-nubank-evolved-its-risk-analysis-platform/)
- Revolut: 4x reduction in card fraud (2018). About 30% precision and 96% recall (press). — [Revolut newsroom](https://www.revolut.com/news/revolut_unveils_new_fleet_of_machine_learning_technology_that_has_seen_a_fourfold_reduction_in_card_fraud_and_had_offers_from_banks_/); [TechHQ](https://techhq.com/?p=192761)
- Wise: per-typology precision tracking is planned. — [Wise 2026](https://wise.jobs/blog/2026-9/keeping-millions-safe-scaling-fincrime-machine-learning-at)
- Base rates for context: Stripe about 1/1,000 payments; Monzo about 1/10,000 transactions. — [Stripe 2023](https://stripe.dev/blog/how-we-built-it-stripe-radar); [Monzo 2025](https://monzo.com/blog/build-a-reactive-fraud-prevention-platform)

### Inferences
- At base rates of 1/1,000 to 1/10,000, accuracy and ROC-AUC alone are misleading. Practitioners report PR-based metrics, false-positive or block rate, and approval rate, and they choose thresholds from cost (margin vs. chargeback fee) rather than from a fixed 0.5.
- Revolut's reported ~30% precision at 96% recall fits Stripe's argument that low precision can still be profitable when the break-even precision is low and the friction is soft (in-app confirmation rather than a hard decline).

### Gaps
- No company published absolute fraud-loss dollars attributable to its ML system, or explicit cost matrices beyond Stripe's worked example.
- No source reported AUC or PR-AUC values for production models.
