# Chapter 4: Training Data

#### Sampling Methods

1. **Nonprobability sampling** selects without probability criteria – creates selection biases but common in practice (language models on Wikipedia/Reddit, self-driving cars from sunny Phoenix/Bay Area).   
2. **Simple random sampling** gives equal probability but misses rare classes.   
3. **Stratified sampling** samples each group separately ensuring rare class inclusion; fails for multilabel tasks.   
4. **Weighted sampling** assigns selection probabilities using domain expertise or to correct distribution mismatch.   
5. **Reservoir sampling** handles streaming data: keep first k elements, then for nth element replace random reservoir position with probability k/n – works when size unknown and can't fit in memory.   
6. **Importance sampling** samples from easy distribution Q(x), reweights by P(x)/Q(x) to approximate expensive P(x); used in reinforcement learning. Note: sample weights (importance during training) differ from weighted sampling (selection).

#### Labeling

**Hand labels** are expensive (subject matter expertise), slow (speech transcription takes 400× duration), and privacy-threatening (can't ship confidential data). Slow iteration when requirements change. 

* **Label multiplicity** from multiple annotators with conflicting labels requires clear problem definition and clear guidelines on labeling.   
* **Data lineage** tracks sample/label origin to flag biases and debug quality issues. 

**Natural labels** are system-evaluated without annotation – recommender clicks (63% of companies), delivery time estimation, stock price prediction. 

* For Natural labels one of critical features is feedback loop length spans prediction to feedback: short (minutes for recommendations), medium (hours for videos), long (weeks for clothing, 1-3 months for fraud disputes). Short windows risk premature negative labels (Twitter: most ad clicks within 5 minutes but some hours later); long loops delay issue detection.

#### Handling Lack of Labels

**Weak supervision** encodes heuristics as labeling functions (LFs): keywords, regex, database lookup, model outputs. Also called programmatic labeling.  
**Semi-supervision** leverages structural assumptions from small initial labels. 

* **Self-training** iteratively adds high-probability predictions to training set.   
* **Similarity-based** assumes similar samples share labels.   
* **Perturbation-based** assumes small noise (white noise, random embeddings) preserves labels. Can match supervised performance despite discarded labels.

**Transfer learning** reuses base model (language modeling on cheap/abundant text) for downstream task.

* Can be both **\-** through *zero-shot* and through *fine-tuning* that continues training with optional prompts. Larger models perform better but cost tens of millions (GPT-3). Lowers entry barriers.

**Active learning** labels most helpful samples via *uncertainty measurement* (lowest probability) or *query-by-committee* (ensemble disagreement). Samples from synthesised regions, pools, or real-time streams. Enables faster adaptation to changing environments.

#### Class Imbalance

Imbalance is the norm – fraud (6.8¢ per $100), spam (85%), disease screening, resume screening (98% eliminated), object detection. Three challenges: insufficient signal for minorities (few-shot or assumes don't exist), nonoptimal heuristics (outputs majority for 99.99% accuracy without learning), asymmetric costs (misclassifying cancer costlier than normal).

**Metrics**: accuracy dominated by majority. Use:

* precision \= TP/(TP+FP)  
* recall \= TP/(TP+FN)  
* F1 \= 2×precision×recall/(precision+recall) – asymmetric, values change with positive class choice. 

ROC plots true positive vs false positive rate; Precision-Recall curve more informative for heavy imbalance.

There are many tecnics to fight the class imbalance. Eg:   
* **Resampling** modifies training distribution (never evaluate on resampled – overfits).   
* **Undersampling** removes majority;   
* **Tomek links** removes majority near minorities.  
* **Oversampling** copies minority;   
* **Two-phase** trains on resampled then fine-tunes on original.   
* **Dynamic sampling** adjusts during training.  
* **Algorithm-level** modifies loss function.  
* **Cost-sensitive** uses matrix Cij; class-balanced weights Wi \= N/(samples in i); focal loss weights by difficulty (lower probability → higher weight).

#### Data Augmentation

Data augmentation is a family of techniques that are used to increase the amount of training data. Augmented data can make our models more robust to noise and even adversarial attacks. **Adversarial attacks** \- using deceptive data to trick a neural network into making wrong predictions.

Three main types of data augmentation:

**Label-preserving transformations** modify without changing labels. Vision: crop, flip, rotate, erase – "computationally free" (CPU transforms while GPU trains). NLP: replace with synonyms or close embeddings that wouldn’t change the meaning or the sentiment of the sentence.

**Perturbation** adds noise. Networks sensitive: changing one pixel misclassifies 67.97% CIFAR-10, 16.04% ImageNet. **Adversarial augmentation** adds noisy samples (random or DeepFool minimum injection \- algorithm that finds the minimum possible noise injection needed to cause a misclassification with high confidence) to improve boundary recognition. BERT randomly replaces 1.5% tokens with random words for small boost. Less common in NLP \- small fraction of random replacement gives model a small performance boost, need to be cariful because of it can create gibberish sentencies.

**Data synthesis** generates samples. NLP templates with slot-filling templates generate thousands. Improves generalisation, reduces memorisation, increases adversarial robustness. In computer vision to synthesize new data \- combine existing examples with discrete labels to generate continuous labels. Neural synthesis (CycleGAN \- generated by other models) actively researched, not yet popular in production.