# Production Fraud/Risk Systems at Marketplace, Ride-Hailing, Delivery, Travel and E-commerce Platforms (incl. Graph-Based Fraud Ring Detection)

Research note scope: Uber, Airbnb, DoorDash, Lyft, Grab, Shopify, Booking.com, eBay, Amazon/AWS, and graph-based fraud ring detection in industry. Research date: 2026-11-25. I read some sources in full (Uber Mastermind, Grab graph service platform, Airbnb ATO newsroom post). Others I know only from search-result excerpts or third-party summaries, and each finding says which. Two important primary posts returned HTTP 403 and could not be read in full: DoorDash's 2026 rules-engine post and Booking.com's graph post. Their findings come from search excerpts.

## 1. What is each company's architecture for real-time risk decisions (ingestion, feature computation, model serving, rules)?

### Takeaway
The pattern repeats across companies. A synchronous "checkpoint" call is made at each risk-relevant user event (login, trip request, booking, checkout). It fetches real-time and precomputed features in parallel, calls a separate ML scoring service, then runs a rules layer that turns scores and features into actions (allow, friction/step-up, hold, reject, ban). Uber (2017) and DoorDash (2022 and 2026) document this most explicitly. Slower, aggregate-level attack detection runs separately as batch or anomaly-detection jobs (Uber RADAR on Spark, DoorDash's Airflow anomaly platform).

### Cited Findings
**Uber**
- Mastermind (published March 2017) is Uber's fraud rules engine. It detects fraud "in a fraction of a second" across payment fraud, account takeover, driver-rider collusion and promotion abuse. — [Uber Engineering: Mastermind](https://www.uber.com/blog/mastermind/)
- Mastermind flow: a request goes to a feature aggregation service, which fetches real-time features (from services) and precomputed features (from data stores, e.g. "number of ride requests in the last hour") in parallel. An ML service then returns risk scores, which are added to the features before the rules execute. — [Uber Mastermind](https://www.uber.com/blog/mastermind/)
- Structure: checkpoints run sets of independent rules. Each rule has properties (per-city/country status and constants), predicates (conditions, combined only with AND at the time) and actions (e.g. reject, ban, require validation). Actions are aggregated into one output, and bans run asynchronously via a fraud actions service. — [Uber Mastermind](https://www.uber.com/blog/mastermind/)
- Performance: running 300 complex rules took about 30 ms. The Python GIL prevents parallel rule execution, so caching parsed syntax trees was key. — [Uber Mastermind](https://www.uber.com/blog/mastermind/)
- Uber covers about two dozen user events. ML models are built only for important, complex events such as trip requests, and rules cover the rest because they react faster than ML can be retrained. — [Uber Mastermind](https://www.uber.com/blog/mastermind/)
- Project RADAR (Feb 2022) is an AI fraud detection and mitigation system with humans in the loop. It monitors fine-grained marketplace segments, detects the start of a fraud attack using time-series anomaly detection (open-source Orbit, time-varying regression, dozens of short- and long-term anomaly signals), and **auto-generates a rule** that fraud analysts review and approve. It runs on Apache Spark and Peloton (batch/scheduled, not per-request). It forecasts "immature" losses (payments not yet settled) as a smart threshold. — [Uber Engineering: Project RADAR](https://eng.uber.com/blog/project-radar-intelligent-early-fraud-detection/) (search excerpt)
- An Uber fraud-prevention team profile describes moving services to real time ("seconds — from minutes and hours"). — [Uber: Fraud prevention team profile](https://www.uber.com/us/en/blog/fraud-prevention-team-profile/) (search excerpt, undated in results)

**DoorDash**
- "Fighting Fraud at Scale: Insights from building a real-time rules engine" (9 March 2026, Fengjiao Jiang and Aditya Upadhyaya). Fraud rules previously lived in application code. The new engine decouples fraud decision logic from application deployments. Each checkpoint owns its rule set and rollout strategy on a shared platform, and rules are readable expressions validated on every change. — [DoorDash Engineering (careersatdoordash.com/?p=15737)](https://careersatdoordash.com/?p=15737); [daily.dev summary](https://app.daily.dev/posts/fighting-fraud-at-scale-insights-from-building-a-real-time-rules-engine-4hozdnao6) (search excerpt only; the page returned 403)
- 2022: DoorDash's anti-fraud rule engine provides fault isolation, logging/metrics observability, data-source integration and microservice integration. It works alongside an ML service, and each evaluation computes "hundreds of complex model features" in real time. — [DoorDash: Ship to production darkly (2022)](https://careersatdoordash.com/blog/ship-to-production-darkly-moving-fast-staying-safe-with-ml-deployments/), via [ZenML summary](https://www.zenml.io/mlops-database/doordash-doordashs-ml-platform-dark-shipping-rollout-for-ml-fraud-detection-models-with-shadow-traffic-fault-isolation-a)
- A 2025 DoorDash post describes an anomaly-detection platform that runs as a daily Airflow job to catch fraud trends developing over days to weeks. A 2022 post covers a shared web library for quickly implementing risk "frictions". — [DoorDash Engineering, via search excerpts](https://careersatdoordash.com/blog/category/engineering/backend/) (I did not fetch the individual posts)
- A DoorDash job posting states the goal of making the rules engine more flexible and scalable and building an "AI-first fraud platform". — [Greenhouse job posting](https://job-boards.greenhouse.io/doordashusa/jobs/7762703) (search excerpt)

**Airbnb**
- 2014 architecture: a separate fraud prediction service computes features. When an event such as a reservation occurs, it queries a scoring service and can place the reservation on hold. Risk types that are not time-critical are scored offline. — [Airbnb Engineering: Architecting a Machine Learning System for Risk (2014)](https://medium.com/airbnb-engineering/architecting-a-machine-learning-system-for-risk-941abbba5a60) (search excerpt)
- A chargeback model is trained on labeled past bookings to estimate each booking's fraud probability. Flagged users get targeted friction (e.g. micro-authorizations), and A/B tests measure each friction's impact on good users. Airbnb absorbs chargeback costs rather than passing them to hosts. — [Airbnb Engineering: Fighting Financial Fraud with Targeted Friction (2018)](https://medium.com/airbnb-engineering/fighting-financial-fraud-with-targeted-friction-82d950d8900e); [InfoQ 2018](https://www.infoq.com/news/2018/03/financial-fraud-ml-airbnb/)
- Airbnb fraud models generally favour recall over precision, and precision/recall need reweighting when class proportions differ between training and evaluation. — [Airbnb Engineering: Designing Machine Learning Models (2015)](https://medium.com/airbnb-engineering/designing-machine-learning-models-7d0048249e69) (search excerpt)
- Airbnb Trust job listings describe "a low-latency framework that intercept[s] all platform requests for risk evaluation purposes, check[s] user capability restrictions and appl[ies] remediation actions." — [Airbnb careers: Staff SWE, Trust & Safety](https://careers.airbnb.com/positions/6943729) (job ad; search excerpt)

**Lyft**
- LyftLearn Serving runs many models on each ride request, including a fraud model that scans the transaction for suspicious activity. Teams such as Fraud Detection get their own repository and microservice. — [ByteByteGo: How Lyft built an ML platform…](https://blog.bytebytego.com/p/how-lyft-built-an-ml-platform-that) (secondary source)
- Lyft splits fraud decisioning between analyst-written business rules and ML models built by research scientists. A convolutional-recurrent neural network reportedly raised flagged fraudulent users by over 40% at the same true-positive rate. — [bestpractice.ai case study](https://bestpractice.ai/ai-use-cases/case-studies/transportation-logistics/lyft-delivered-a-40-increase-in-potentially-fraudulent-users-detected-without-increasing-false-positives-by-using-neural-networks) (secondary; not verified against Lyft's own posts)
- Lyft has an InfoQ talk on ML with streaming. — [InfoQ: ML streaming at Lyft](https://www.infoq.com/presentations/ml-streaming-lyft) (not reviewed)

**Shopify**
- Shopify's fraud model scores each payment's risk in real time. It was trained on over 10 billion transactions across millions of merchants. — [Shopify blog: best-in-class technology protects merchants from fraud](https://www.shopify.com/blog/shopify-best-in-class-technology-protects-millions-of-merchants-from-fraud) (company marketing)
- Card-testing defence (Oct 2023): a proprietary ML model scores every payment attempt before it reaches the processor. Shopify claims it blocks 90% of card-testing attacks and gives a 13% authorization-rate lift. At the late-2022 peak, attackers ran up to about 1M card-enumeration attempts per week. — [Shopify Enterprise: Block card testing attacks](https://shopify.com/enterprise/blog/block-card-testing-attacks) (company marketing)
- Shopify reports a 99.7% acceptance rate and "$745M+ fraudulent activity flagged". — [Shopify fraud control](https://www.shopify.com/nz/fraud-control) (marketing)

**Booking.com**
- Booking.com runs a real-time graph service that feeds its fraud models. The fraud system screens all incoming requests during request processing, so graph generation must be low latency. — [Booking.com Engineering: Leverage graph technology for real-time fraud detection and prevention](https://medium.com/booking-com-development/leverage-graph-technology-for-real-time-fraud-detection-and-prevention-438336076ea5) (search excerpt; the page returned 403)
- Booking.com researchers tested whether tabular Transformers beat GBDTs for fraud detection (arXiv 2024, revised 2025). — [arXiv 2405.13692](https://arxiv.org/abs/2405.13692)
- A 2021 Booking.com paper frames ML in anti-fraud operations, including how models fit investigators' daily work. — [arXiv 2107.01979](https://arxiv.org/pdf/2107.01979v1)
- At MRC Barcelona 2023, Booking.com described a multi-stage ML system that monitors the behaviour of newly onboarded hotels to detect fraudulent listings. — [MRC: behavioral features talk](https://merchantriskcouncil.org/learning/resource-center/events/barcelona/uncovering-the-hidden-gems-of-fraud-detection-leveraging-behavioral-features)

### Inferences
- The consensus architecture is checkpoint → parallel feature fetch (real-time and precomputed) → ML score as a feature → rules produce actions → async enforcement. It maps directly onto a Kafka → feature service → model → decision-engine pipeline.
- Uber and DoorDash both run two loops. The synchronous loop handles per-event decisions in milliseconds. The asynchronous loop handles attack and trend detection over hours to weeks, and can feed new rules back into the synchronous loop (RADAR auto-generates rules).

### Gaps
- I could not read DoorDash's 2026 rules-engine post in full (403), so its latency, throughput and expression-language specifics are unverified.
- I found no official Lyft engineering post specifically on its fraud architecture.
- Shopify (Fraud Protect), Amazon retail and Etsy have published no detailed engineering architecture that I found. Shopify's material is marketing.
- I did not research Meta or LinkedIn anti-abuse within the tool budget.

## 2. How do they detect account takeover (ATO) specifically: device fingerprinting, login risk, behavioral signals, step-up authentication?

### Takeaway
The publicly documented approach is a login/action risk model. It estimates the probability that the actor is the true owner, using device, IP, geo and velocity signals, and triggers step-up verification (OTP to a new device, extra confirmation) when risk is high. Airbnb is the clearest primary source. Uber lists ATO as a Mastermind use case but gives no model details.

### Cited Findings
- Airbnb (13 April 2017): an ML model predicts "the probability that each login or action on Airbnb is being performed by the true account owner". It was trained on hundreds of millions of historical login events labelled good or bad, and weighs hundreds of signals: unexpected country, unfamiliar IP/computer/phone, an unusually high number of logins from one IP. — [Airbnb Newsroom: What we're doing to prevent account takeovers](https://news.airbnb.com/prevent-account-takeovers/)
- On high predicted ATO risk, Airbnb requires additional confirmation. New devices must be verified by one-time code (SMS or email) and are then remembered. Alerts were expanded to SMS and to more account-change types. Airbnb admitted the model had not stopped every takeover. — [Airbnb Newsroom](https://news.airbnb.com/prevent-account-takeovers/); [Engadget 2017](https://www.engadget.com/2017-04-15-airbnb-multi-factor-authentication.html)
- Uber Mastermind covers account takeover among its checkpoints. Actions include "require validation" (step-up), slowing request velocity for moderate-risk users, and reject/ban for high-risk users. — [Uber Mastermind](https://www.uber.com/blog/mastermind/)
- Grab: GrabDefence's graph maps users, networks (e.g. IP addresses) and devices. One use case is detecting multiple accounts sharing one physical device. — [Grab Engineering: Graph service platform (Jan 2023)](https://engineering.grab.com/graph-service-platform); [GrabDefence and AWS](https://www.grab.com/sg/business/defence/grabdefence-and-aws/)
- DoorDash built a shared web library for quickly implementing risk frictions (2022). — [DoorDash Engineering blog index (search excerpt)](https://careersatdoordash.com/blog/category/engineering/backend/)

### Inferences
- Graduated response is standard: allow, then friction/step-up (OTP, micro-auth, re-verification), then hold/reject, then ban. Actions are tied to score bands rather than a binary block, which reduces false-positive cost to good users (Airbnb A/B-tests frictions).
- Device novelty, IP novelty or volume, and geo mismatch are the canonical ATO features, and they are typically computed as per-user "first seen" and velocity aggregates.

### Gaps
- I found no recent (2022–2026) primary engineering post from Airbnb, Uber or DoorDash detailing ATO model features, behavioural biometrics or device-fingerprinting internals.

## 3. How do they build and serve velocity/aggregate features at low latency (Flink, Kafka Streams, Redis, feature stores)?

### Takeaway
Velocity features are computed in streaming jobs (Flink at DoorDash) or precomputed into low-latency stores (Redis at DoorDash), then fetched in parallel at decision time (Uber's feature aggregation service). Precise latency numbers are rarely published. Uber's 30 ms for 300 rules is the main concrete figure, and it covers rule execution only.

### Cited Findings
- DoorDash built real-time feature pipelines as native Flink applications serving features from a Redis-backed store. These pipelines were later judged a bottleneck for faster feature development. — [DoorDash Engineering (careers.doordash.com/?p=3828)](https://careers.doordash.com/?p=3828) (search excerpt)
- DoorDash real-time features improved ETA predictions, with a real-time Dasher wait-time signal among the top features (non-fraud example of the same infra). — [DoorDash Engineering (?p=1393)](https://careersatdoordash.com/?p=1393) (search excerpt)
- Uber Mastermind fetches real-time features and precomputed store values (e.g. ride requests in the last hour) in parallel, and slows the "ride request velocity" of moderate-risk users. — [Uber Mastermind](https://www.uber.com/blog/mastermind/)
- DoorDash's fraud models compute hundreds of features in real time. Shadow rollout checks training/inference feature consistency, missing data and timeouts. — [ZenML summary of DoorDash 2022](https://www.zenml.io/mlops-database/doordash-doordashs-ml-platform-dark-shipping-rollout-for-ml-fraud-detection-models-with-shadow-traffic-fault-isolation-a)
- Grab ingests streaming graph writes through Kafka (Confluent) into the graph service, partly to reduce load on Neptune. — [Grab Engineering: Graph service platform](https://engineering.grab.com/graph-service-platform)
- Stripe's "Shepherd" adapts Airbnb's open-source Chronon feature platform for ML feature development, relevant as an industry feature-store pattern for fraud. — [Stripe blog: Shepherd](https://stripe.com/blog/shepherd-how-stripe-adapted-chronon-to-scale-ML-feature-development) (not fetched)
- Kai Waehner (Confluent) blogs on Kafka/Flink fraud detection in mobility (e.g. FREE NOW, Grab). — [kai-waehner.de](https://www.kai-waehner.de/?p=7516) (vendor-affiliated; not fetched)

### Inferences
- The "DoorDash Riviera" name in the brief did not appear in any search result, so I could not tie it to fraud. Treat the brief's reference as unverified. What is supported is that DoorDash used native Flink with a Redis store for real-time features.
- Train/serve feature consistency is a first-class production concern, which DoorDash verifies explicitly in shadow mode.

### Gaps
- I found no published p99 latency numbers for feature retrieval at Uber, DoorDash, Airbnb or Lyft fraud systems.
- I did not verify current (2024–2026) DoorDash or Uber feature-store details (e.g. Uber Palette/Michelangelo for fraud).

## 4. How is graph analysis used for fraud rings, and is it real-time or batch?

### Takeaway
Industry uses three tiers of graph analysis:
- Real-time neighbourhood lookups on shared entities (device, IP, card), via graph DB OLTP (Grab on Neptune, Booking.com real-time graph service).
- Batch GNNs (RGCN, heterogeneous GNNs) whose risk scores become features for the main real-time model (Uber, eBay xFraud, Grab).
- Batch community detection or anomaly detection for ring discovery and investigation (Grab GraphBEAN, AWS Neptune Analytics + GraphStorm).

GNN inference is mostly batch or near-real-time. Real-time graph use is mostly lookups and simple link features.

### Cited Findings
- Uber (May 2021) used a relational graph convolutional network (RGCN) on a driver–rider graph, with edges from shared information and multiple edge types, to detect collusion. Labels were chargebacks in a time window. The resulting user fraud scores were fed as features into the downstream production risk model. Adding two such score features gave **15% better precision with minimal increase in false positives**. The post states the model "is not used in Uber's production platform". — [Uber Engineering: Using Relational Graph Learning to Detect Collusion](https://eng.uber.com/fraud-detection/)
- Grab applies RGCN for fraud detection across its ecosystem. It argues that graph models need less feature engineering and fewer labels, and that they help detect unknown fraud. — [Grab Engineering: Graph for fraud detection](https://engineering.grab.com/graph-for-fraud-detection) (search excerpt)
- Grab GraphBEAN is an unsupervised autoencoder on a bipartite graph of GrabFood/GrabMart transactions that scores both nodes and edges for anomalies. — [Grab Engineering: Graph anomaly model](https://engineering.grab.com/graph-anomaly-model) (search excerpt)
- Grab graph service platform (5 Jan 2023) has four layers: storage (S3 raw files, **Amazon Neptune** graph, DynamoDB metadata/schema), drivers, service, and REST APIs. The Search API serves real-time OLTP for streams, and the Analysis API serves batch OLAP. Ingestion is batch CSV from S3 via ETL with schema validation, plus streaming via Kafka. Use cases include multiple accounts sharing a device, transaction loops and money laundering. No latency or scale figures are given. — [Grab Engineering: Graph service platform](https://engineering.grab.com/graph-service-platform)
- Grab has a graph visualisation platform for investigators ("10X investigation"). — [Grab Engineering: Graph visualisation](https://engineering.grab.com/graph-visualisation)
- Grab and AI Singapore research targets acting on fraud within hundreds of milliseconds on million-scale graphs. This is stated as a research goal, not a production claim. — [Grab: A glimpse into the cutting edge of anti-fraud research](https://www.grab.com/inside-grab/stories/a-glimpse-into-the-cutting-edge-of-anti-fraud-research/)
- Booking.com runs real-time graph generation inside request-time fraud screening (low latency required). It presented graph technology for payment fraud ("fraudster webs") at MRC Vegas 2023. — [Booking.com Engineering (Medium)](https://medium.com/booking-com-development/leverage-graph-technology-for-real-time-fraud-detection-and-prevention-438336076ea5) (excerpt); [MRC Vegas 2023 talk](https://merchantriskcouncil.org/learning/resource-center/events/vegas/fraud-networks-our-answer-to-detecting-fraudster-webs)
- eBay xFraud is an explainable heterogeneous-GNN transaction-fraud detector for eBay's real-world e-commerce environment, with open-source code. — [arXiv 2011.12193 (xFraud)](https://arxiv.org/pdf/2011.12193)
- Delivery Hero uses Amazon Neptune for fraud detection (AWS case study, vendor). — [AWS Database Blog: Delivery Hero with Amazon Neptune](https://aws.amazon.com/blogs/database/empowering-fraud-detection-at-delivery-hero-with-amazon-neptune) (not fetched)
- AWS: Neptune ML automates GNN training (DGL) on graph data in Neptune. AWS Solutions guidance builds a heterogeneous graph from tabular data and trains a GNN for near-real-time transaction fraud. A newer pattern uses **Neptune Analytics + GraphStorm** to enrich the graph with GNN embeddings and risk scores, then applies **community detection** to surface rings. — [AWS: Neptune Analytics + GraphStorm fraud](https://aws.amazon.com/blogs/database/use-graph-machine-learning-to-detect-fraud-with-amazon-neptune-analytics-and-graphstorm); [AWS Guidance: near-real-time GNN fraud](https://docs.aws.amazon.com/solutions/near-real-time-fraud-detection-with-graph-neural-network-on-aws/); [AWS: real-time fraud with Neptune ML](https://aws.amazon.com/blogs/database/build-a-real-time-fraud-detection-solution-using-amazon-neptune-ml); [re:Invent 2022 DAT301](https://d1.awsstatic.com/events/Summits/reinvent2022/DAT301_Use-graphs-to-perform-fraud-analysis-with-Amazon-Neptune-and-Neptune-ML.pdf) (vendor)
- **Amazon Fraud Detector closed to new customers on 7 Nov 2025.** AWS points to SageMaker, AutoGluon and AWS WAF as alternatives. The docs say "no longer open to new customers", not shut down. — [AWS: Amazon Fraud Detector availability change](https://docs.aws.amazon.com/frauddetector/latest/ug/frauddetector-availability-change.html)
- GrabDefence partnered with Amazon Fraud Detector. — [GrabDefence and AWS](https://www.grab.com/sg/business/defence/grabdefence-and-aws/) (now possibly affected by the AFD closure)
- Neo4j claims graph databases uncover e-commerce fraud patterns "in real time". This is vendor marketing. — [Neo4j blog](https://neo4j.com/blog/fraud-detection/graph-database-ecommerce-fraud); [Neo4j whitepaper](https://go.neo4j.com/rs/710-RRC-335/images/Neo4j_WP-Fraud-Detection-with-Graph-Databases.pdf)
- An academic paper deploys an end-to-end risk-control system on TigerGraph/GSQL (not eBay). — [arXiv 2101.01898](https://arxiv.org/pdf/2101.01898)

### Inferences
- A practical, evidence-backed design for a streaming system works in two stages. (a) In real time, maintain shared-entity links (card/device/email/IP → accounts) and compute cheap features such as the "number of accounts sharing this device" or connected-component size. (b) In batch, run GNN or community detection and push scores back as features. Uber's result (+15% precision from two GNN score features) supports the "graph score as feature" approach over end-to-end real-time GNN inference.
- Vendor claims of "real-time" GNN fraud detection (AWS, Neo4j) should be weighed against the company sources, which mostly use batch GNNs.

### Gaps
- I found no published latency or scale numbers for Grab's or Booking.com's real-time graph queries.
- I found no evidence that eBay or Etsy use TigerGraph or Neo4j in production. I found nothing on Etsy fraud ML.
- I could not confirm Neptune ML's current deprecation status.

## 5. How do they let analysts write/deploy rules quickly, and how are rules and ML combined?

### Takeaway
Rules are separated from application code into a rules engine with a safe expression language, a UI, and evaluate/shadow modes before activation. This brings deploy time from about an hour (or a full app release) down to about a minute. ML scores enter the rules as just another feature, so analysts set thresholds and actions per segment. New models roll out via shadow traffic, then an A/B test.

### Cited Findings
- Uber Mastermind v1 used Python dictionaries, and each rule change needed engineers and took about one hour. v2 introduced "Predicate-eval", a free-form rule language built on Python's AST with a whitelist, which avoids the risks of raw `eval()` and blocks unsafe comparisons like `None < 1`. Analysts write rules in a front end, test in **Evaluate** mode (logged, no user impact), then switch to **Active**. Changes reach production in about one minute. v1 accumulated 300 rules over about 1.5 years. After the v2 front end, analysts wrote 200 new rules in three months. v2 took four months to build and two months to migrate, with shadow testing on production traffic. — [Uber Mastermind (2017)](https://www.uber.com/blog/mastermind/)
- Uber noted that rules decay over weeks as fraudsters adapt, and planned automatic rule-effectiveness evaluation plus routing low-confidence cases to manual review. — [Uber Mastermind](https://www.uber.com/blog/mastermind/)
- Uber RADAR closes the loop by auto-generating rules from detected anomalies, with analyst approval. RADAR's write-up stresses that explainability is "of paramount importance" in fraud detection. — [Uber RADAR (2022)](https://eng.uber.com/blog/project-radar-intelligent-early-fraud-detection/) (search excerpt)
- DoorDash (2026) decouples rules from deployments, gives each checkpoint its own rule set and rollout strategy, and validates readable rule expressions on every change. **Backtesting** replays recorded facts through the same evaluation engine to produce a counterfactual ("what decision would have been made had a rule existed then"). **Shadow mode** evaluates rules on live traffic without applying outcomes. — [DoorDash Engineering 2026](https://careersatdoordash.com/?p=15737) (search excerpt)
- DoorDash ML rollout ("dark shipping", 2022) runs in three stages: shadow at 1% (errors, missing data, timeouts, feature consistency), shadow at 100% (performance and load), then a live A/B test in Curie, champion vs challenger, with full traffic only after a statistically significant win. The rule engine provides fault isolation so a bad model cannot cascade. — [DoorDash 2022](https://careersatdoordash.com/blog/ship-to-production-darkly-moving-fast-staying-safe-with-ml-deployments/) via [ZenML](https://www.zenml.io/mlops-database/doordash-doordashs-ml-platform-dark-shipping-rollout-for-ml-fraud-detection-models-with-shadow-traffic-fault-isolation-a)
- Lyft reportedly splits analyst business rules from ML models. — [bestpractice.ai](https://bestpractice.ai/ai-use-cases/case-studies/transportation-logistics/lyft-delivered-a-40-increase-in-potentially-fraudulent-users-detected-without-increasing-false-positives-by-using-neural-networks) (secondary)
- Airbnb A/B-tests each friction on good users, tying ML scores to graduated actions. — [InfoQ 2018](https://www.infoq.com/news/2018/03/financial-fraud-ml-airbnb/)

### Inferences
- Several features recur across sources and are strong design references for a FYP decision engine: rules as data with a sandboxed expression language, per-checkpoint rule sets, evaluate/shadow modes, backtesting by replaying logged features, and the ML score treated as an input feature to rules.
- Logging the full "facts" (feature vector) at decision time is what enables backtesting. DoorDash's design relies on it.

### Gaps
- I did not obtain DoorDash's 2026 rule DSL syntax, storage backend or latency.
- I found no primary-source detail on Airbnb's, Grab's or Booking.com's analyst rule tooling.
