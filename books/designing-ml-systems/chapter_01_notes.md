# Chapter 1: Overview of Machine Learning Systems 

**An ML system includes**: business requirements, user interface, data stack, model development, monitoring, updating logic, and infrastructure. The algorithm is a small part of the system.

Instead of hand-coding rules (traditional software), you provide inputs and outputs and let the system **learn patterns**. This works when patterns are too complex to manually specify but simple enough for machines to learn.

Batching improves throughput at the cost of latency. Processing 50 queries in 20ms gives 2,500 queries per second vs. 100 queries per second when processing individually at 10ms. By accepting higher latency through batching, you can **significantly improve throughput**.

**Latency distribution matters more than averages.** Use percentiles (p90, p95, p99), not mean values. One 3,000ms outlier among nine 100ms requests produces a misleading 390ms average. The slowest users often matter most. A 100ms delay can reduce revenue by 7%. A 30% latency increase can drop conversion by 0.5%. More than 3 seconds of load time can cause over 50% of mobile users to leave.

Traditional software engineering focuses on code testing. In ML systems, datasets must be versioned, data quality must be tested, and data poisoning must be prevented. 

**Problem framing matters more than model choice**. You must define the prediction target, constraints, success metric (offline and online), and trade-offs early; otherwise you optimize the wrong objective.

Research and production follow fundamentally different approaches:

**Research approach:**

- Optimize for single objective (accuracy on benchmarks)  
- Use ensembles and complex techniques for small improvements  
- Fast training, high throughput priority  
- Work with static, clean datasets  
- Fairness and interpretability optional  
- Competition winners (Kaggle, Netflix Prize)

**Production approach:**

- Balance multiple conflicting objectives (latency, revenue, user experience)  
- Simple models unless complexity justifies significant gains  
- Fast inference, low latency priority  
- Handle messy, shifting data constantly  
- Fairness and interpretability mandatory  
- 100ms delay \= 7% revenue loss

Most ML jobs require production skills. Techniques that win competitions are rarely used in real systems. If you optimize accuracy first and add fairness later, it is often too late, because ML systems encode biases at scale.

When patterns change constantly, ML is superior to rules. For static problems like mapping zip codes to states, lookup tables are enough. For evolving problems like spam detection, ML adapts without rewriting rules.
