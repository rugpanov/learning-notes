# Chapter 5: Feature Engineering

#### Learned vs Engineered Features

Deep learning has automated much of feature engineering for text and images – instead of manual lemmatisation, stopword removal, or image feature extraction, you tokenise raw text or input raw images directly and let the model learn representations. 

However, most production ML systems need features beyond raw text/images: user metadata, interaction counts, temporal patterns, contextual signals.

#### Common Feature Engineering Operations

**Missing values** come in three types: 

* **MNAR** (missing not at random – missingness depends on the true value, e.g. high earners not disclosing income);  
* **MAR** (missing at random – depends on another observed variable, e.g. gender A not disclosing age);  
* **MCAR** (missing completely at random – no pattern, very rare in practice). Two strategies exist:  
- **Deletion** – column (remove variable) or row (remove sample). Row deletion only works for MCAR with \<0.1% missing. Both risk losing information or introducing bias  
- **Imputation** – fill with defaults, mean, median, or mode. Avoid filling with possible values (0 for number of children) – makes it impossible to distinguish missing from actual zeros. No perfect method: deletion risks bias, imputation risks noise and data leakage

**Feature scaling** gets features into similar ranges. Three approaches: **min-max scaling** to \[0, 1\] or \[–1, 1\] (empirically better), **standardisation** (zero mean, unit variance) for normally distributed features, and **log transformation** to reduce skewness. Scaling requires global statistics (min, max, mean) computed from training data only – reuse these at inference. Neglecting scaling can cause gibberish predictions, especially with gradient-boosted trees and logistic regression.

**Discretisation** (binning) turns continuous features into buckets (e.g. income brackets). Helps with limited training data but introduces discontinuities at boundaries. Rarely helpful in practice.

**Encoding categorical features** is straightforward for static categories but problematic in production where categories change constantly (new brands, accounts, domains). The **hashing trick** (Vowpal Wabbit) hashes categories into a fixed-size space – handles unseen categories automatically. Collisions are random rather than systematic; even 50% collision rate causes \<0.5% performance loss (Booking.com research). Essential for continual learning settings.

**Feature crossing** combines two or more features to model nonlinear relationships (e.g. marital status × number of children). Essential for linear models, logistic regression, and tree-based models; occasionally helps neural networks learn faster (DeepFM, xDeepFM). Caveat: 100 × 100 possible values \= 10,000 combinations – increases overfitting risk and data requirements.

**Positional embeddings** encode sequence position for models like transformers that process tokens in parallel. Two approaches: **learned** (embedding matrix with position columns, summed with word embeddings – used in BERT) and **fixed** (sine/cosine functions, a special case of **Fourier features**). Fixed embeddings generalise to continuous coordinates (3D object surfaces) where learned embedding matrices can't.

#### Data Leakage

Data leakage occurs when label information "leaks" into training features but isn't available at inference. Common causes:

- **Time-correlated splits** – randomly splitting time-dependent data leaks future information. Always split by time: train on first N periods, split the last period into validation/test  
- **Scaling before splitting** – computing global statistics on entire dataset before splitting leaks test distribution into training. Always split first, then scale using train statistics only  
- **Missing value imputation** – using mean/median from entire dataset instead of train split only  
- **Data duplication** – same samples appearing in train and test splits. CIFAR-10 had 3.3% test duplicates in training set (discovered 10 years after release). Always check for duplicates before splitting; oversample after splitting  
- **Group leakage** – correlated samples (e.g. two CT scans of same patient a week apart) split across train/test  
- **Data generation process** – e.g. hospital sending suspected cancer patients to different scan machines, model learns machine type instead of cancer signs

Detection strategies: measure feature-label correlation (investigate unusually high values), run **ablation studies** (significant performance drop when removing a feature warrants investigation), monitor new features carefully – dramatic improvement may indicate leakage rather than quality.

#### Engineering Good Features

More features generally improves performance, but too many features cause data leakage opportunities, overfitting, increased serving memory/latency, and technical debt. L1 regularisation should theoretically zero out useless features, but removing them explicitly helps models learn faster.

Two evaluation criteria for feature quality:

- **Feature importance** – measured via XGBoost's built-in functions or model-agnostic **SHAP** (SHapley Additive exPlanations), which shows both global importance and per-prediction contribution. Facebook's ads team found top 10 features account for \~50% of total importance; last 300 features contribute \<1%  
- **Feature generalisation** – assessed via **coverage** (percentage of samples with values present) and **distribution overlap** between train and test splits. Trade-off between generalisability and specificity: IS\_RUSH\_HOUR generalises better than HOUR\_OF\_THE\_DAY but loses information
