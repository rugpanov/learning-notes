# Chapter 6: Model Development and Offline Evaluation

#### Model Selection

Model selection starts by matching algorithm families to task type (text classification → logistic regression, RNNs, transformers; anomaly detection → k-NN, isolation forest). Beyond accuracy/F1/log loss, evaluate data requirements, training cost, inference latency, and interpretability.

Six selection tips:

1. **Avoid the SOTA trap**: academic benchmarks don't predict performance on your data or latency constraints.

2. **Start with the simplest model** to validate pipelines, ease debugging, and set a baseline – pretrained BERT is low effort to start but high effort to improve upon.

3. **Avoid human bias**: engineers invest disproportionately in preferred architectures, so give each candidate comparable experiment budgets.

4. **Evaluate now vs. later**: build learning curves by training on cumulative data buckets (batch_1 ⊂ batch_2 ⊂ batch_3) and plotting accuracy vs. data volume. If the curve has plateaued, more data won't help; tree-based models may dominate early while neural networks overtake them once data doubles. Diminishing returns: the last 20% of quality requires disproportionate effort.

5. **Evaluate trade-offs**: false positives/negatives, compute/accuracy, performance/interpretability.

6. **Understand model assumptions**: IID (neural networks), smoothness, tractability P(Z|X) (generative models), linear boundaries (linear classifiers), conditional independence (naive Bayes).

**Four phases of ML adoption** – each phase's output serves as the next phase's baseline:

1. non-ML heuristics ("a heuristic will get you 50% of the way there", Zinkevich);
2. simplest ML model to validate the full pipeline;
3. optimise via hyperparameter search, feature engineering, ensembles;
4. complex models only after simple models plateau.

#### Ensembles

Ensembles boost performance consistently – 20/22 Kaggle winning solutions in 2021 used them. Low correlation among base learners is the key driver; mixing a transformer, RNN, and GBT beats three similar models. Ensembles are less common in production due to deployment complexity but remain standard where small gains yield large financial impact (ad CTR).

Three classical ensemble methods; MoE as a modern extension:

1. **Bagging** creates independent bootstraps via sampling-with-replacement and combines predictions by majority vote or average. Reduces variance; random forests use bagging with feature randomness.

2. **Boosting** trains learners iteratively, reweighting misclassified samples so subsequent learners focus on harder examples; final prediction is a weighted combination. XGBoost and LightGBM (parallel learning, faster on large datasets) are standard. Doesn't work well with neural networks – NNs overfit the training set, so passing "hard examples" forward is pointless.

3. **Stacking** trains a meta-learner (logistic regression or majority vote heuristic) on base learner outputs. A weak model in the ensemble (0.7 accuracy when others are 0.9) can drag performance down – monitor each model's contribution.

4. **Mixture of Experts (MoE)** is its own paradigm: specialised sub-models + a gating network that directs each input to one or more experts; in classical MoE only one expert is active at a time. Used in modern LLMs. **Test-time scaling** applies the same idea at inference: run the model N times with temperature > 0, then select or aggregate the best answer.

#### Experiment Tracking, Versioning and Debugging

**Isolated experiment principle**: only one parameter should change between two compared experiments – otherwise you can't attribute the performance difference to any cause.

Track per experiment: loss curves (train/eval splits), performance metrics, sample/prediction/label logs, throughput, memory/GPU utilisation, learning rate schedule, gradient norms. Tools: TensorBoard → Weights & Biases → ClearML/MLflow (historical progression). Track one key metric deeply rather than many superficially – excess metrics dilute attention. Minimum viable set: loss (train + val), target quality metric (train + val), one system metric.



**Data versioning** is the "flossing" of ML: everyone agrees it's important, few do it. Datasets are too large for line-by-line diffs, rollback copies are unfeasible, merge conflict semantics are unclear, and GDPR may mandate deletion. DVC registers diffs only on checksum change or file add/remove. Full reproducibility also requires environment tracking: CUDA atomic operations introduce nondeterminism between runs.

**Debugging** is hard because ML fails silently (code compiles, loss decreases, predictions are wrong), validation requires full retraining (hours), and root cause spans data/labels/features/code/infrastructure across teams. Five failure sources: theoretical constraints (wrong model family), implementation bugs, poor hyperparameters, data problems (mismatched labels, stale statistics), poor feature selection. Three proven techniques: 

1. **start simple and add components incrementally** rather than cloning SOTA;

2. **overfit a single batch** – if 10 images can't reach 100% accuracy, the implementation is broken; 

3. **set a random seed** to isolate model changes from stochastic variation.

#### Distributed Training

**Gradient checkpointing** trades compute for memory: 10x larger models on the same GPU at 20% extra computation.

**Data parallelism** replicates the model and splits data across workers. 

* **Synchronous SGD** waits for all gradients – stragglers stall the system as worker count grows. 

* **Asynchronous SGD** updates immediately but risks **gradient staleness**; mitigated in practice when updates are sparse since gradient updates rarely collide on the same parameters (Hogwild!). Large-scale parallelism inflates effective batch size (GPT-3: 3.2M); learning rate scaling is unstable past a threshold and yields diminishing returns.

**Model parallelism** splits model layers across machines; sequential layer dependencies cause pipeline stalls. 

**Pipeline parallelism** breaks mini-batches into micro-batches so machines overlap forward and backward passes. Both methods are often combined for better hardware utilisation.

#### AutoML

**Hyperparameter tuning** (soft AutoML) is the most common production form, covering learning rate, batch size, layer sizes, dropout, optimiser betas, and quantisation precision. Tuned weaker models can outperform stronger architectures (Melis et al. 2018). Use validation split for tuning; never the test split. Methods: random search, grid search, Bayesian optimisation.

**Neural architecture search** (hard AutoML) treats architecture components as hyperparameters. Requires: a search space (building blocks and constraints), a performance estimation strategy (avoid training all candidates to convergence), and a search strategy (RL or evolutionary algorithms).

**Learned optimisers** replace hand-designed optimisers (Adam, SGD) with neural networks trained on thousands of aggregated tasks; they generalise to new datasets and architectures and can train improved versions of themselves. EfficientNets achieve SOTA accuracy at up to 10x better efficiency. Upfront cost is prohibitive for most organisations, but resulting architectures and optimisers lower production cost broadly.

#### Baselines and Offline Evaluation

Metrics mean nothing without baselines. Five types:

1. **random baseline** (uniform or label-distribution – a 90/10 imbalanced task yields F1≈0.10 with label-distribution random despite 82% accuracy); 

2. **simple heuristic** (chronological ranking, frequency-based recommendations); 

3. **zero rule** (always predict the majority class); 

4. **human baseline** (essential when automating human decisions); 

5. **existing solutions** (current if/else logic or third-party models). Distinguish a *good* system (beats prior art) from a *useful* system (meets real-world adoption threshold).

Six evaluation methods beyond aggregate accuracy:

1. **Perturbation tests** add noise to test data (background noise, clipping, typos) to approximate real-world variance; prefer models that perform best on perturbed data – sensitivity predicts fragility and adversarial vulnerability. 

2. **Invariance tests** change sensitive attributes (race, gender, name) while holding everything else constant; output changes indicate bias. 

3. **Directional expectation tests** verify outputs change in the expected direction (larger lot → higher house price). 

4. **Model calibration** ensures predicted probability X occurs X% of the time; miscalibration distorts expected-value estimates (click volume, recommendation diversity). Measure with a calibration curve; calibrate post-hoc with Platt scaling. Nate Silver: calibration is "the single most important test of a forecast." 

5. **Confidence measurement** sets a per-prediction usefulness threshold rather than an aggregate metric, routing uncertain predictions to humans or requesting more information. 

6. **Slice-based evaluation** examines performance on data subgroups – overall metrics hide disparate treatment (96.2% overall with 80% minority vs. 95%/95%). **Simpson's paradox** makes slicing critical: aggregate trends can reverse within subgroups. Find slices via domain heuristics, error analysis on misclassified examples, or automated slice finders (beam search, clustering, decision trees).
