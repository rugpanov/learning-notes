# Chapter 2: Introduction to Machine Learning Systems Design

ML systems must be motivated by business objectives, not ML metrics. Companies care about revenue, conversion rates, and user satisfaction—not accuracy improvements from 94% to 94.2%. If you optimise ML metrics without moving business metrics, your project will be killed. Try to **map ML metrics to business metrics early.**

Use A/B testing to validate the relationship between ML and business performance. Sometimes the relationship is impossible to trace. In complex systems, **ML is just one component.**  Also problem framing matters more than algorithms. A business problem ("speed up customer support") must be translated into an ML problem with clear inputs, outputs, and objective function.

**ML returns compound over time.** In 2020, 75% of mature ML adopters (5+ years) deployed models in under 30 days, while 60% of beginners took over 30 days. Mature pipelines develop faster, need less engineering time, and have lower cloud bills.

**ML systems require four core characteristics:**

- **Reliability**: Continue performing correctly despite failures. ML systems fail silently—no crashes or errors, just wrong predictions users may not notice. If you translate text to a language you don't know, you can't tell if Google Translate is wrong.  
- **Scalability**: Handle growth in complexity (logistic regression to 100M-parameter neural network), traffic volume (10K to 10M requests), and model count (1 to 8,000 models). Requires both resource scaling (autoscaling GPUs) and artifact management (automated monitoring/retraining for hundreds of models). Even Amazon failed at autoscaling on Prime Day—one hour of downtime cost $72-99 million.  
- **Maintainability**: Different contributors (ML engineers, DevOps, SMEs) work together using their preferred tools. Code, data, and artifacts must be versioned and reproducible so other contributors have context when original authors leave.  
- **Adaptability**: Discover performance improvements and update without service interruption as data distributions and business requirements shift.

**Developing ML systems is iterative, not linear.** Expect to retrain multiple times—wrong labels, class imbalance (99.99% negative), stale data, or changing objectives (impressions to click-through rate).

**Task types and their challenges:**

- **Binary classification** (2 classes) is simpler than **multiclass** (3+ classes). High cardinality (1,000+ classes) requires hierarchical classification and at least 100 examples per class—1,000 classes means 100,000+ examples minimum.  
- **Multilabel classification** (examples belong to multiple classes) is the hardest: annotators disagree on how many labels apply, and extracting predictions from probabilities is ambiguous. Given \[0.45, 0.2, 0.02, 0.33\], do you pick top 2 or top 3?  
- **Classification vs regression**: Often interchangeable. Regression becomes classification by quantizing outputs into buckets. Classification becomes regression by outputting probabilities with a threshold.

Reframe problems to make them easier. Predicting which app a user opens next: bad framing \= multiclass (output distribution over N apps, retrain when new app added); better framing \= regression (output score for each app given user/context, no retraining needed for new apps).

**Common loss functions exist for standard tasks.** For regression: RMSE (root mean squared error) or MAE (mean absolute error). Being 10 minutes late once vs 2 minutes late 5 times: MAE treats them equally (10 total), but RMSE punishes the single big error more (10² vs 5×2²). Use RMSE when big mistakes are unacceptable, MAE when all errors matter equally. For binary classification: logistic loss (measures how far predicted probability is from 0 or 1). For multiclass: cross entropy (measures how different predicted probability distribution is from actual labels—if model predicts \[0.45, 0.2, 0.02, 0.33\] but truth is \[0, 0, 0, 1\], cross entropy quantifies that gap). Most engineers use these defaults rather than designing custom objective functions.

**Decouple multiple objectives into separate models.** When optimizing conflicting goals (engagement vs quality), combining losses into one model (loss \= α × quality\_loss \+ β × engagement\_loss) requires retraining every time you tune weights. Better: train separate models, combine outputs (α × quality\_score \+ β × engagement\_score). Adjust priorities without retraining, and update objectives independently since spam techniques evolve faster than quality perception.

**Data vs algorithms debate remains unsettled.** Judea Pearl: "Data is profoundly dumb. ML folks who follow the data-centric paradigm will be outdated in 3-5 years." Christopher Manning: "Huge computation with massive data and simple algorithms create incredibly bad learners. Structure lets us learn more from less data." Richard Sutton's bitter lesson: "70 years of AI research shows general methods leveraging computation are most effective." Peter Norvig: "We don't have better algorithms. We just have more data."

**Data is essential but not sufficient.** Monica Rogati's data science hierarchy: data lies at the foundation—without data collection, storage, and pipelines, ML is impossible. Recent ML progress relies on data scale: One Billion Word Benchmark (0.8B tokens), GPT-2 (10B tokens), GPT-3 (500B tokens). However, more low-quality data (outdated, mislabeled) can hurt performance.  