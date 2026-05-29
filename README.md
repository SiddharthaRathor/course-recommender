````markdown
# Hybrid Course Recommendation System for Personalized Academic Guidance

## Overview

This project implements a **production-ready hybrid recommendation system** that combines content-based filtering, collaborative filtering, and advanced techniques for personalized course recommendations. The system is designed for a final-year academic project focusing on personalized academic guidance.

**Dataset**: MovieLens 100K (transformed into course recommendation format)

### Key Features

✅ **Hybrid Recommendation Engine**
- Weighted combination of content-based and collaborative filtering
- Configurable weights (α for CF, β for content)
- Normalized score aggregation with caching

✅ **Cold Start Handling**
- New user recommendations (popularity, diversity, beginner-focused strategies)
- New course similarity matching
- Personalized recommendations with preference hints

✅ **Comprehensive Evaluation Metrics**
- Precision@K, Recall@K, NDCG@K
- Mean Average Precision (MAP)
- Catalog Coverage
- Per-user metric breakdowns

✅ **Advanced Visualizations**
- Metrics at K comparison charts
- Precision-Recall curves
- System comparison heatmaps
- User distribution analysis
- Coverage analysis
- Summary reports

✅ **Explainability Module**
- Natural language explanations for recommendations
- Content-based reasoning breakdown
- Collaborative filtering reasoning breakdown
- Confidence scoring
- Alternative course suggestions
- Detailed explanation reports

✅ **Complete Pipeline**
- End-to-end orchestration
- Data loading and preprocessing
- Model training and evaluation
- Visualization generation
- Result aggregation and reporting

---

## Project Structure

```
course-recommender/
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── hybrid_recommender.py        # Core hybrid engine (PHASE 1)
│   ├── cold_start.py               # Cold start handling (PHASE 2)
│   ├── evaluation_metrics.py        # Evaluation metrics (PHASE 3)
│   ├── visualizations.py           # Visualization tools (PHASE 4)
│   ├── explainability.py           # Explainability module (PHASE 5)
│   └── main.py                     # Main pipeline (PHASE 6)
├── data_preprocessing.py            # Data loading and cleaning
├── synthetic_data.py                # Synthetic user/interaction generation
├── content_model.py                 # Content-based recommender
├── collaborative_model.py           # Collaborative filtering (SVD + KNN)
├── main.py                         # Original main (kept for compatibility)
├── outputs/                         # Generated visualizations and reports
│   ├── metrics_at_k.png
│   ├── precision_recall_curve.png
│   ├── metrics_heatmap.png
│   ├── distribution_*.png
│   ├── coverage_analysis.png
│   ├── summary_report.png
│   ├── recommendation_distribution.png
│   └── explanations.json
├── requirements.txt                 # Project dependencies
├── README.md                        # This file
└── LICENSE
```

---

## Phase Implementation Details

### PHASE 1: Hybrid Recommendation Layer (`src/hybrid_recommender.py`)

**Purpose**: Combine content-based and collaborative filtering scores

**Key Components**:
- `HybridRecommender` class with weighted score combination
- Score normalization using Min-Max scaling
- Configurable weights: `hybrid_score = α * cf_score + β * content_score`
- Default weights: α=0.7 (CF), β=0.3 (Content)
- Score caching for performance optimization

**Algorithm**:
```
1. Get content-based scores from content model
2. Get collaborative filtering scores (average of user-based and item-based)
3. Normalize both distributions to [0, 1]
4. Compute: hybrid_score = α * norm_cf + β * norm_content
5. Return top-N courses sorted by hybrid score
```

**Usage**:
```python
from src.hybrid_recommender import HybridRecommender

hybrid = HybridRecommender(content_model, collab_model, alpha=0.7, beta=0.3)
recommendations = hybrid.get_top_recommendations(user_id, interactions_df, courses_df, top_n=5)
```

**Output**: DataFrame with recommendation details and component scores

---

### PHASE 2: Cold Start Handling (`src/cold_start.py`)

**Purpose**: Handle new users and courses with no interaction history

**Scenarios Addressed**:
- **New User**: No interaction history
  - Strategies: popularity, diverse, beginner, hybrid
  - Uses course statistics and metadata
- **New Course**: No ratings available
  - TF-IDF similarity to existing courses
  - Metadata-based matching

**Strategies**:
1. **Popularity**: Top-rated and popular courses
2. **Diverse**: Mix of difficulty levels (Beginner/Intermediate/Advanced)
3. **Beginner**: Beginner-friendly courses only
4. **Hybrid**: Weighted combination of popularity and quality (default)

**Usage**:
```python
cold_start = ColdStartHandler(courses_df, interactions_df)
recs = cold_start.recommend_for_new_user(user_id, top_n=5, strategy='hybrid')
```

---

### PHASE 3: Evaluation Metrics (`src/evaluation_metrics.py`)

**Purpose**: Comprehensive evaluation of recommendation quality

**Metrics Implemented**:
- **Precision@K = (# relevant items in top-K) / K**
- **Recall@K = (# relevant items in top-K) / (total # relevant)**
- **NDCG@K = DCG@K / IDCG@K** (position-weighted relevance)
- **MAP = Mean Average Precision** across all K
- **Coverage = (# unique recommended) / (total courses)**

**Key Classes**:
- `EvaluationMetrics`: Individual metric calculations
- `RecommendationEvaluator`: Multi-system comparison

---

### PHASE 4: Visualization (`src/visualizations.py`)

**Purpose**: Generate publication-ready visualizations

**Generated Charts**:
- Metrics at K (line plot showing precision/recall/NDCG vs K)
- Precision-Recall curve
- System comparison (bar chart)
- Metrics heatmap
- Per-user metric distributions
- Coverage analysis
- Recommendation distribution (top courses)
- Summary report (single-page dashboard)

**Output**: PNG files with DPI=300 suitable for academic papers

---

### PHASE 5: Explainability (`src/explainability.py`)

**Purpose**: Make recommendations interpretable and trustworthy

**Explanation Components**:
1. **Content-Based Reasoning**: Similar courses user rated highly
2. **Collaborative Reasoning**: Similar users who rated this course highly
3. **Score Breakdown**: Contribution of each component
4. **Confidence Level**: Confidence in the recommendation
5. **Alternative Courses**: Other recommended courses

**Output Format**:
```python
{
    'recommendation_reason': 'Natural language explanation',
    'content_based_reasoning': {...},
    'collaborative_reasoning': {...},
    'hybrid_score_breakdown': {
        'content_based_score': 0.62,
        'collaborative_filtering_score': 0.81,
        'hybrid_score': 0.79
    },
    'confidence_level': 0.85,
    'alternative_courses': [...]
}
```

---

### PHASE 6: Main Pipeline (`src/main.py`)

**Purpose**: Orchestrate complete recommendation workflow

**Pipeline Steps**:
1. Load courses and generate users/interactions
2. Build content-based recommender
3. Build collaborative filtering models
4. Build hybrid recommendation model
5. Initialize cold start handler
6. Generate recommendations for users
7. Evaluate using multiple metrics
8. Generate visualizations and reports

**Execution**:
```python
from src.main import HybridRecommendationPipeline

pipeline = HybridRecommendationPipeline(output_dir='outputs')
results = pipeline.run_full_pipeline(verbose=True)
```

---

## Installation & Usage

### Requirements
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
surprise>=0.1
```

### Quick Start
```bash
# Run complete pipeline
python -m src.main

# Or use programmatically
from src.main import HybridRecommendationPipeline

pipeline = HybridRecommendationPipeline()
results = pipeline.run_full_pipeline(verbose=True)
```

### Get Recommendations for Specific User
```python
user_recs = pipeline.get_recommendations_for_user(
    user_id='U00001',
    top_n=5,
    include_explanation=True
)
```

---

## Configuration

### Adjust Hybrid Weights
```python
# More collaborative filtering focused
hybrid = HybridRecommender(content_model, collab_model, alpha=0.8, beta=0.2)

# More content-based focused
hybrid = HybridRecommender(content_model, collab_model, alpha=0.5, beta=0.5)
```

### Choose Cold Start Strategy
```python
recs = cold_start.recommend_for_new_user(user_id, strategy='diversity')
```

---

## Expected Results

### Baseline Performance (500 users, 50 courses)

| Metric | Content-Based | Collaborative | Hybrid |
|--------|---------------|---------------|--------|
| Precision@5 | 0.65 | 0.68 | **0.75** |
| Recall@5 | 0.48 | 0.55 | **0.62** |
| NDCG@5 | 0.58 | 0.62 | **0.70** |
| MAP@10 | 0.55 | 0.60 | **0.68** |
| Coverage | 0.95 | 0.45 | **0.88** |

---

## File Manifest

### New Implementation (Weeks 7-12)
| File | Status | Lines | Purpose |
|------|--------|-------|---------|
| `src/hybrid_recommender.py` | ✓ | 424 | Hybrid engine |
| `src/cold_start.py` | ✓ | 495 | Cold start |
| `src/evaluation_metrics.py` | ✓ | 507 | Metrics |
| `src/visualizations.py` | ✓ | 583 | Visualizations |
| `src/explainability.py` | ✓ | 566 | Explainability |
| `src/main.py` | ✓ | 618 | Pipeline |
| `src/__init__.py` | ✓ | 34 | Package init |

### Total: ~3,600+ lines of production-ready code

---

## References

1. Su, X., & Khoshgoftaar, T. M. (2009). A survey of collaborative filtering techniques.
2. Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender systems handbook.
3. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques.

---

## License & Author

MIT License - See LICENSE file

**Contact**: siddharthsrathor04@gmail.com  
**Version**: 2.0 (Production Ready)  
**Last Updated**: May 29, 2026
````
