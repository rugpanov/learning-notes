# Chapter 11: The Human Side of Machine Learning

#### User Experience

ML predictions are inherently uncertain, so UX must communicate confidence and handle failure gracefully. Two key principles: **smooth failing** (when predictions are wrong, the cost to the user should be low) and **showing multiple predictions** rather than a single answer – autocomplete suggesting 3–5 options lets users pick correctly even when the top prediction is wrong.

Users anchor on ML outputs even when warned they may be wrong. In high-stakes domains (medical diagnosis, criminal justice), incorrect predictions can be harder to override than having no prediction at all – **automation bias** makes humans defer to the system. Design should make it easy to dismiss or override ML suggestions rather than requiring active effort to reject them.

#### Team Structure

ML projects require diverse roles: **ML engineers** (model development), **data engineers** (pipelines and infrastructure), **data scientists** (analysis and experimentation), **DevOps/MLOps** (deployment and monitoring), and **product/domain experts** (problem framing and evaluation). Miscommunication between these roles is a leading cause of project failure.

Two organisational models: **centralised ML team** serving the whole company (ensures consistency, avoids duplicated work, but becomes a bottleneck) vs **embedded ML engineers** within product teams (faster iteration, better domain understanding, but fragmented tooling and duplicated infrastructure). Most mature organisations converge on a **hybrid**: a central platform team providing shared infrastructure with embedded engineers in product teams.

End-to-end data scientists – who handle everything from data to deployment – are rare and expensive. Companies that expect one person to do the full pipeline typically underinvest in infrastructure, leading to burnout and brittle systems.

#### Responsible AI

ML systems can cause harm through **bias** (systematic disadvantage to certain groups), **lack of transparency** (users and affected parties can't understand decisions), and **privacy violations** (models memorising or leaking training data).

**Bias** enters at every stage: historical bias in training data, representation bias (undersampled groups), measurement bias (proxies that correlate with protected attributes), and aggregation bias (one model for diverse subpopulations). Mitigation strategies span pre-processing (rebalancing data), in-processing (fairness constraints during training), and post-processing (adjusting thresholds per group). No single fairness metric satisfies all definitions simultaneously – **impossibility theorem**: calibration, false positive parity, and false negative parity cannot all hold unless prevalence is equal across groups.

**Interpretability** exists on a spectrum. **Intrinsically interpretable** models (linear regression, short decision trees) are transparent by design. **Post-hoc methods** explain black-box models: **SHAP** (per-feature contribution), **LIME** (local linear approximation), **attention visualisation** (what the model "looks at"). Trade-off: more complex models are harder to interpret, but interpretability tools are improving faster than complexity is growing.

**Privacy** concerns include models memorising training examples (extractable via adversarial queries), feature stores aggregating sensitive user data, and prediction APIs leaking information about the training distribution. Techniques: **differential privacy** (adding calibrated noise to guarantee individual records can't be identified), **federated learning** (training on-device without centralising data), and **data minimisation** (collect only what's needed, delete when no longer required).

#### A Framework for Responsible ML

Four practices for teams:

1. **Discover sources of harm early** – red-teaming, bias audits, stakeholder interviews before deployment
2. **Establish accountability** – clear ownership of model behaviour, documented decision-making processes, and incident response plans
3. **Provide recourse** – affected users must have a way to contest automated decisions and reach a human reviewer
4. **Iterate continuously** – responsible AI is not a one-time audit but ongoing monitoring, evaluation, and improvement as data and society evolve
