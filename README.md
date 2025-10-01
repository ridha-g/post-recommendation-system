# Post Recommendation System

A Neural Collaborative Filtering (NCF) system that recommends personalized content by learning from user interests, engagement patterns, and post attributes. Built with PyTorch and achieves **54.5% Precision@3**, significantly outperforming traditional baseline methods.

## Problem Statement

Given a social media platform with users and posts, recommend the top 3 most relevant posts for each user based on:
- User profile interests (e.g., "sports", "technology", "literature")
- Past engagement behavior (likes/dislikes)
- Post content attributes (tags, content type)

**Challenge**: Limited interaction data (20 engagements per user on average) with cold-start issues for new users and posts.

## Approach

### Architecture
Hybrid Neural Collaborative Filtering model combining:
- **Collaborative Filtering**: 32-dimensional user and post embeddings to capture latent preferences
- **Content-Based Features**: TF-IDF vectors (200 features) encoding user interests and post tags
- **Deep Fusion Network**: Multi-layer perceptron with dropout regularization

```
Inputs: [User_Embedding(32) + Post_Embedding(32) + User_TF-IDF(200) + Post_TF-IDF(200)]
        ↓
Engagement Weighting: features × (1 + 0.5 × past_engagement_norm)
        ↓
MLP: Dense(464→64, ReLU, Dropout) → Dense(64→32, ReLU) → Dense(32→1, Sigmoid)
        ↓
Output: Engagement Probability [0,1]
```

### Key Innovation
The network automatically learns optimal weighting between collaborative and content signals for each user-post pair, discovering that highly engaged users respond more to content-type preferences while less active users rely on explicit interest matches.

## Results

<img width="772" height="150" alt="image" src="https://github.com/user-attachments/assets/4049393b-becf-40cf-bc1a-882d22e7e87c" />


**Training**: Converges in 10 epochs (0.53s) from loss 0.695 → 0.565  
**Inference**: 0.63s total runtime for 50 users

### Sample Recommendations

**User U11** (interests: gaming, literature, tech | engagement: 0.85)
1. P88 (score: 0.721) - Literature text post → Direct interest match
2. P70 (score: 0.714) - Art + food image → Learned visual preference  
3. P29 (score: 0.706) - Sports + music audio → Diversified content

## Dataset

- **Users**: 50 users with age, gender, top 3 interests, past engagement scores
- **Posts**: 100 posts with creator, content type (text/image/video/audio), tags
- **Engagements**: 1,000 binary interactions (49.7% engagement rate)

Located in `data/` directory.

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/post-recommendation-system.git
cd post-recommendation-system

# Install dependencies
pip install -r requirements.txt
```

**Requirements**: Python 3.8+, PyTorch 2.0+, scikit-learn, pandas, numpy, matplotlib

## Usage

### Quick Start (Jupyter Notebook)
```bash
jupyter notebook recommendation_system.ipynb
```

### Run as Script
```bash
python recommendation_system.py
```

The script will:
1. Load user, post, and engagement data
2. Train the NCF model (10 epochs)
3. Generate top-3 recommendations for each user
4. Save outputs to `outputs/recommendations.csv`

### Output Format
```csv
user_id,rank,recommended_post_id,score
U1,1,P70,0.732961
U1,2,P78,0.731526
U1,3,P12,0.718602
```

## Project Structure

```
post-recommendation-system/
├── recommendation_system.ipynb    # Main implementation notebook
├── recommendation_system.py       # Python script version
├── report.pdf                     # Technical report
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── data/
│   ├── Users.csv                  # User profiles
│   ├── Posts.csv                  # Post metadata
│   └── Engagements.csv            # Interaction history
└── outputs/
    ├── recommendations.csv        # Generated recommendations
    ├── training_curves.png        # Loss curves
    └── model_comparison.png       # Performance comparison chart
```

## Methodology

I tested multiple approaches before arriving at the final solution:

1. **Jaccard Similarity Baseline** - Simple interest-tag overlap, but missed latent patterns
2. **Logistic Regression** - Fast but couldn't capture non-linear relationships
3. **Basic NCF** - Good embeddings but ignored rich content metadata
4. **Final Hybrid NCF** - Combines collaborative + content features for best performance

See `report.pdf` for detailed analysis of each approach.

## Challenges & Limitations

### Challenges Faced

1. **Data Sparsity**: With only 20 engagements per user on average, building reliable user profiles was difficult. Many user-post pairs had no interaction history.

2. **Cold Start Problem**: New users with <5 interactions and new posts with no engagement history posed significant challenges for the embedding-based approach.

3. **Class Imbalance**: Nearly balanced dataset (49.7% engagement rate) was actually challenging - neither class dominated, requiring careful model tuning.

4. **Cross-Validation Trade-off**: With only 1,000 samples, splitting into folds reduced training data too much (667 samples/fold), causing underfitting. Had to choose between robust validation and sufficient training data.

5. **Feature Engineering**: Deciding how to combine collaborative and content signals required experimentation - simple concatenation vs weighted combinations vs learned interactions.

### Current Limitations

1. **No Temporal Context**: The model doesn't consider when engagements happened. Recent posts aren't prioritized over older ones, and user preferences may have changed over time.

2. **Static Recommendations**: Once trained, the model doesn't update with new interactions. In production, this would require periodic retraining or online learning.

3. **Limited Diversity Control**: While diversity score is high (0.899), there's no explicit constraint preventing all recommendations from being the same content type or creator.

4. **No Explainability**: The model is a "black box" - while it makes good predictions, explaining WHY a specific post was recommended requires manual inspection of features.

5. **Scalability Concerns**: Current implementation scores all user-post pairs (50 × 100 = 5,000 predictions). For 10K users and 100K posts, this becomes 1 billion predictions, requiring optimization (approximate nearest neighbors, candidate generation).

6. **Binary Engagement Only**: The model only knows like/dislike, not engagement intensity (shares, comments, time spent). Multi-level engagement would provide richer training signal.

7. **No Contextual Factors**: Doesn't consider time of day, device type, user location, or social context that might affect preferences.

### Potential Issues

- **Popularity Bias**: The model might favor popular posts that many users engaged with, potentially under-recommending quality niche content.
- **Filter Bubble Risk**: Without explicit diversity constraints, users might get stuck seeing similar content repeatedly.
- **Data Privacy**: User interests and engagement patterns are sensitive data requiring proper handling in production.

## Future Enhancements

- **Temporal decay**: Weight recent posts higher with exponential decay
- **Diversity constraints**: MMR algorithm to prevent filter bubbles
- **Multi-task learning**: Jointly predict engagement type (like/share/comment)
- **Attention mechanisms**: Explainable recommendations with attention weights
- **Production deployment**: Redis caching + FAISS indexing for <20ms latency

## Technical Details

**Model**: PyTorch implementation of Neural Collaborative Filtering  
**Training**: Adam optimizer (lr=0.001), Binary Cross-Entropy loss, 80/20 train-test split  
**Regularization**: Dropout (0.2) + L2 weight decay (1e-5)  
**Evaluation**: Precision@K, Recall@K, NDCG@K for ranking quality

## References

- He, X., et al. (2017). "Neural Collaborative Filtering." WWW 2017.
- TF-IDF vectorization for semantic content matching
- PyTorch for deep learning implementation

## Contact

For questions or collaboration opportunities, reach out via ridhagulzar@gmail.com.

---

**License**: MIT  
**Author**: Ridha Mohammadi  
**Last Updated**: October 2025
