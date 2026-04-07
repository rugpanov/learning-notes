# Chapter 10: Infrastructure and Tooling for MLOps

#### Infrastructure Layers

ML infrastructure sits on four layers, each abstracting the one below:

1. **storage and compute** (where data is stored and computations run)

2. **resource management** (scheduling and orchestrating workloads)

3. **ML platform** (model development, training, deployment tooling)

4. **development environment** (IDE, versioning, CI/CD).

Smaller companies typically adopt managed cloud services; larger organisations build internal platforms once standardisation benefits outweigh the engineering cost.

#### Storage and Compute

The storage layer handles data across the ML lifecycle – training data, model artefacts, logs, features. **HDD** is cheap for bulk archival; **SSD** for latency-sensitive serving. Cloud object stores (S3, GCS) dominate due to durability and elastic scaling, but network latency matters for training workloads – local NVMe or distributed file systems (HDFS) reduce I/O bottleneck during data-intensive training.

Compute separates into **training** (GPU/TPU clusters, high memory, tolerant of minutes-long latency) and **inference** (optimised for low latency, cost per prediction). CPUs remain viable for inference at moderate scale; GPUs are necessary when batch throughput or model size demands it. Cloud instances are priced by the second – **spot/preemptible instances** cut GPU costs 60–90% but can be reclaimed, so checkpointing is essential. Multi-tenancy (sharing GPUs across teams) improves utilisation but requires careful scheduling.

#### Development Environment

Standardised dev environments prevent "works on my machine" failures. Three approaches: 

1. **container-based** (Docker images with pinned dependencies – most common), 

2. **cloud-based IDEs** (Jupyter on remote clusters, convenient but can encourage monolithic notebooks), 

3. **infrastructure-as-code** (Terraform, Pulumi for reproducible provisioning).

Notebooks are popular for experimentation but notoriously difficult to version, test, and review. Production-grade teams typically prototype in notebooks, then refactor into modular Python packages with proper testing. 

**Dev/prod parity** is critical – differences in library versions, hardware, or data access between development and production are a leading source of deployment failures.

#### Resource Management

As ML workloads scale, manual resource allocation fails. Four tools handle this at increasing abstraction:

- **Cron** – simplest scheduler; triggers jobs at fixed times. No dependency awareness, no retry logic, no resource management. Adequate for single daily batch jobs
- **Orchestrators** (Airflow, Prefect, Dagster) – manage DAGs of dependent tasks with retry, alerting, and backfill. Airflow is the de facto standard but has a steep learning curve; newer tools (Prefect, Dagster) simplify configuration. Orchestrators handle workflow logic, not resource allocation
- **Compute resource managers** (Kubernetes, YARN, Mesos) – schedule containers across clusters, handle resource requests/limits, auto-scaling, and fault tolerance. **Kubernetes** has become dominant; most ML platforms assume it. Key concepts: pods (smallest unit), nodes (machines), namespaces (isolation), resource quotas (preventing one team from monopolising GPUs)
- **ML-specific schedulers** (Kubeflow, Ray) – layer ML workflow abstractions (distributed training, hyperparameter tuning, serving) on top of Kubernetes. **Ray** handles distributed Python natively; useful for scaling from laptop to cluster without rewriting code

#### ML Platform

The ML platform layer provides higher-level abstractions for the ML lifecycle. Three core components:

**Model store** (or model registry) tracks trained models with metadata: hyperparameters, training data version, metrics, lineage. Enables comparison across experiments, rollback to prior versions, and audit trails. Without a registry, teams lose track of which model is deployed and why.

**Feature store** centralises feature computation, storage, and serving. Two interfaces: **offline** (batch features for training, backed by a data warehouse) and **online** (low-latency features for inference, backed by Redis or DynamoDB). Solves three problems: feature reuse across teams (avoids redundant computation), **train/serve skew** (same transformation logic for training and inference), and feature discovery (catalogue of available features with documentation). Feast and Tecton are common open-source and commercial options.

**Model deployment** tools handle packaging, serving, scaling, and traffic management. Spectrum ranges from simple Flask/FastAPI wrappers to managed platforms (SageMaker, Vertex AI) to Kubernetes-native serving (KServe, Seldon). Key capabilities: **canary deployments** (route a percentage of traffic to the new model), **shadow deployments** (run new model in parallel without serving predictions), A/B testing integration, and auto-scaling based on request volume.

#### Build vs Buy

The build-vs-buy decision recurs at every layer. Managed services (SageMaker, Vertex AI, Databricks) reduce operational burden but create vendor lock-in and limit customisation. Open-source tools offer flexibility but require internal expertise to operate and maintain.

Four decision factors: 

1. **Stage of the company** (startups should buy to move fast; scale-ups build when managed services become the bottleneck), 

2. **Data sensitivity** (regulated industries may need on-premise control), 

3. **Team expertise** (building infrastructure you can't maintain is worse than vendor lock-in), 

4. **Cost trajectory** (cloud costs compound – what's cheap at 10 requests/second may be prohibitive at 10,000). 

The pragmatic path: start with managed services, build only when you hit a hard constraint that the vendor can't solve. "Use, not build, for competitive advantage" – most ML value comes from models and data, not infrastructure.
