# Chapter 3: Data Engineering Fundamentals

#### Data Sources & Data Format

**Data has various sources and different processing requirements.** 

* **User input data** is often malformatted and requires heavy validation plus fast processing (users expect immediate results).  
* **System-generated logs** are well-formatted and can be processed periodically, but volume grows rapidly – store in low-access storage when debugging isn't needed.   
* Also, we can buy **third-party data** (demographic, browsing, purchase history), but it faces privacy restrictions: Apple's opt-in IDFA significantly reduced available data, forcing focus on first-party data.

Tabular data can be stored in row-major or column-major format, each optimised for different access patterns:

- **Row-major (CSV)**: consecutive row elements stored together; faster for writes (adding examples) and row access, but must read all columns then filter when needing only some. NumPy ndarray is row-major by default.  
- **Column-major (Parquet)**: consecutive column elements stored together; faster for column reads – reads 4 out of 1,000 features directly without loading others. pandas DataFrame is column-major, making row iteration 34x slower than column iteration (2.41s vs 0.07s).

**Text files are human-readable but wasteful; binary files are compact.** Storing 1,000,000 requires 7 bytes in text (7 characters) but 4 bytes in binary (int32). Converting CSV to Parquet reduced 14MB to 6MB. AWS reports Parquet is 2x faster to unload and uses 6x less storage than text.

#### Data Models

**Relational vs NoSQL models suit different problems.** 

* Relations are unordered tuples queried with **SQL (declarative language)** – you specify outputs wanted, not retrieval steps. Query optimisers figure out fastest execution, making normalised joins feasible.  
* **Document model** (JSON/BSON) has better **locality** (all book info in one document) but joins across documents are harder and less efficient.   
* **Graph model excels** when relationships are priority: finding everyone born in USA traverses "within" and "born\_in" edges easily, but this query is nearly impossible in SQL with unknown hop counts between country and person.

#### Data Storage Engines

**OLTP vs OLAP are converging as storage and processing decouple.** 

* Transactional databases (OLTP) handle individual transactions with low latency, high availability, and **ACID guarantees** (atomicity, consistency, isolation, durability).   
* Analytical databases (OLAP) handle aggregations across columns efficiently. 

Historically these required separate systems, but modern databases like CockroachDB (transactional with analytical queries) and DuckDB (analytical with transactional queries) bridge the gap. Decoupling storage from compute (BigQuery, Snowflake) allows same data to be processed differently without duplication.

**ETL transforms then loads; ELT loads then transforms.** ETL (extract, transform, load) validates and transforms data before loading into structured storage – more upfront work but easier to query later. ELT (extract, load, transform) dumps raw data into data lakes for fast arrival, then processes on demand—flexible but inefficient to search massive raw data. Hybrid data lakehouse solutions (Databricks, Snowflake) combine flexibility of lakes with management of warehouses.

#### Dataflow modes

Three dataflow modes between processes have distinct trade-offs. 

* Passing through **databases** is simple but slow (unsuitable for strict latency requirements) and requires shared access.   
* Passing through services uses **REST or RPC requests**—request-driven, synchronous, good for logic-heavy systems. If service is down, requests fail or timeout.   
* Passing through **real-time transports** **(Apache Kafka, RabbitMQ)** uses event-driven architecture where services broadcast events to a broker – asynchronous, prevents bottlenecks when hundreds of services need data from each other. Pubsub retains data temporarily (7 days) before deletion; message queues route messages to intended consumers.

**Batch processing** computes static features; **stream processing** computes dynamic features. 

* Historical data in databases is processed in periodic batch jobs (daily) using MapReduce or Spark to extract features that change slowly (driver ratings).  
* Streaming data in real-time transports is processed as it arrives (every minute or on-demand) to extract features that change quickly (available drivers, recent ride requests, median price of last 10 rides). Stream processing enables **stateful computation**: for 30-day trial engagement, compute only new data daily and join with older computation instead of recomputing all 30 days. Apache Flink and KSQL provide SQL abstraction for complex stream queries with hundreds or thousands of features.

Batch processing is a special case of stream processing – easier to make stream processor do batch than vice versa.