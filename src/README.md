# FUSION-SPARK: Real-Time Geo-Spatial Sentiment Monitoring Framework

## Overview

This repository implements the distributed big data framework proposed in the manuscript:

**"A Data-Driven Modeling and Performance Analysis of a Distributed Big Data for Real-Time Geospatial Sentiment Monitoring of Technology Trends."**

The system integrates:

- Apache Flume for high-throughput Twitter ingestion  
- Apache Spark Streaming for distributed micro-batch processing  
- HDFS for fault-tolerant storage  
- Random Forest for sentiment classification  
- Kernel Density Estimation (KDE) for spatial intensity modeling  

The pipeline enables near real-time monitoring of emerging technology discussions across cities.

---

##  Key Capabilities

1. Real-time Twitter ingestion (~1,500 tweets/sec)
2. Spark micro-batch processing (5-second window)  
3. Distributed execution on multi-node cluster  
4. RF sentiment classifier (F1 ≈ 0.91)  
5. Spatial KDE heatmaps for technology trends  
6. Strong and weak scaling validation  
7. Statistical significance testing  
8. Resource utilization profiling  


--- yaml

## System Configuration

### Cluster Setup

| Component | Version |
|----------|---------|
| Apache Spark | 3.x |
| Apache Flume | 1.11 |
| Hadoop HDFS | 3.x |
| Python | 3.9 |
| scikit-learn | ≥1.3 |
| Google Colab RAM | 32 GB |

### Hardware Used

- 5-node Spark cluster  
- 32 total cores  
- 128 GB aggregate memory  

---

## Twitter API Configuration

Create environment variables:

```bash
export TWITTER_API_KEY=xxxx
export TWITTER_API_SECRET=xxxx
export TWITTER_ACCESS_TOKEN=xxxx
export TWITTER_ACCESS_SECRET=xxxx

How to Run
Step 1 — Install dependencies
pip install -r requirements.txt

Step 2 — Start Flume agent
flume-ng agent \
  -n TwitterAgent \
  -f deployment/flume.conf

Step 3 — Submit Spark job
bash deployment/spark_submit.sh

Step 4 — Run main notebook

Open: main.ipynb

Methodology Summary
Micro-batch Formation

The incoming Twitter stream is modeled as:

S(t) = {x₁, x₂, …, xₙ}


Tweets are grouped into micro-batches:

B_k = {x_i | t ∈ [kΔt, (k+1)Δt)}


where:

Δt = 5 seconds

k = batch index

Ingestion Throughput
T = N / Δt


Observed:

T ≈ 1,500 tweets/sec

Random Forest Configuration
Parameter	Value
Trees	200
Max depth	20
Classes	3
Training size	5,000 labeled tweets

Experimental Results
1. Performance

Metric	    Value
Throughput	    1,500 tweets/sec
Micro-batch     latency	3.2 sec
RF F1-score	    0.91
Dataset size	1.6M tweets
Geo-tagged	    ~10%

2. Baseline Comparison
Model	            Accuracy	    F1
Naïve Bayes	        0.82	        0.79
Logistic Regression	0.86	        0.84
SVM	                0.88	        0.87
RF (Proposed)	    0.92	        0.91

Scalability Validation

The repository includes:
Strong scaling analysis
Weak scaling analysis
Parallel efficiency curves
Executor CPU/memory profiling

Statistical Validation

We validate significance using:
Paired t-test
Wilcoxon signed-rank test
McNemar test
All tests show p < 0.05, confirming robustness.

Spatial Analysis

KDE is used to estimate sentiment intensity:

f(x) = (1 / (nh)) Σ K((x − x_i)/h)

This generates smooth geo-heatmaps of technology trends across Indian cities.

Citation

If you use this work, please cite:

@article{fusion_spark_geo_sentiment,
  title={A Data-Driven Modeling and Performance Analysis of a Distributed Big Data for Real-Time Geospatial Sentiment Monitoring of Technology Trends},
  author={Amit Pimpalkar},
  journal={Under Review},
  year={2026}
}
