# Geometric Visualization of Contrastive Learning

This guide explains how to visualize decision boundaries and learned representations in a geometric way.

## Overview

When you train a contrastive learning model, it learns to map images into a high-dimensional **embedding space** (e.g., 2048 dimensions for ResNet50). In this space:

- **Similar images** (same wood type) are placed **close together**
- **Different images** (different wood types) are placed **far apart**

By projecting these high-dimensional embeddings into 2D or 3D, you can visualize:
1. How well the model separates different classes
2. Decision boundaries between classes
3. Which samples are "hard" (near the boundary) vs "easy" (far from boundary)
4. The overall structure and clustering of your data

---

## Visualization Methods

### 1. t-SNE (t-Distributed Stochastic Neighbor Embedding)

**What it does**: Non-linear dimensionality reduction that preserves local structure.

**Best for**:
- Visualizing clusters and groups
- Seeing which samples are neighbors in embedding space
- Understanding local relationships

**How to run**:
```bash
python scripts/visualize_embeddings.py \
    --checkpoint outputs/contrastive/best_model.pth \
    --data_dir data/ \
    --encoder resnet50 \
    --methods tsne \
    --output_dir outputs/visualizations
```

**What to look for**:
- ✅ **Good**: Classes form tight, well-separated clusters
- ❌ **Bad**: Classes overlap significantly or are scattered

**Example interpretation**:
```
Good t-SNE plot:
    Class 0 (Oak)  ●●●●●
                   ●●●●●
                              Class 1 (Pine)  ○○○○○
                                             ○○○○○
    → Classes are well-separated!

Bad t-SNE plot:
    Class 0  ●○●●○
    Class 1  ○●○○●
    → Classes overlap, model struggles to separate them
```

---

### 2. UMAP (Uniform Manifold Approximation and Projection)

**What it does**: Similar to t-SNE but faster and preserves both local AND global structure.

**Best for**:
- Large datasets (faster than t-SNE)
- Preserving overall data topology
- More stable results across runs

**How to run**:
```bash
# First install UMAP
pip install umap-learn

# Then visualize
python scripts/visualize_embeddings.py \
    --checkpoint outputs/contrastive/best_model.pth \
    --data_dir data/ \
    --methods umap \
    --output_dir outputs/visualizations
```

**Advantage over t-SNE**:
- UMAP better preserves the global structure of your data
- If you see multiple clusters in UMAP, they likely represent meaningful subgroups

---

### 3. PCA (Principal Component Analysis)

**What it does**: Linear dimensionality reduction that finds directions of maximum variance.

**Best for**:
- Understanding which features are most important
- Quick visualization (very fast)
- Seeing linear separability

**How to run**:
```bash
python scripts/visualize_embeddings.py \
    --checkpoint outputs/contrastive/best_model.pth \
    --data_dir data/ \
    --methods pca \
    --output_dir outputs/visualizations
```

**What the percentages mean**:
```
PC1 (65% variance) → First component explains 65% of variation
PC2 (20% variance) → Second component explains 20% of variation

High percentages (>60% for PC1+PC2) = Data is more linearly separable
Low percentages (<40% for PC1+PC2) = Data needs non-linear methods
```

---

### 4. Cosine Similarity Matrix

**What it does**: Shows how similar every pair of samples is in embedding space.

**Best for**:
- Seeing fine-grained relationships
- Identifying confusing samples
- Checking if same-class samples are similar

**How to run**:
```bash
python scripts/visualize_embeddings.py \
    --checkpoint outputs/contrastive/best_model.pth \
    --data_dir data/ \
    --methods similarity \
    --output_dir outputs/visualizations
```

**How to read**:
```
Heatmap colors:
🔴 Red (1.0)    = Very similar (identical or same class)
🟡 Yellow (0.0) = Somewhat similar
🔵 Blue (-1.0)  = Very different (opposite direction in embedding space)

Ideal pattern (when sorted by class):
┌───────┬───────┐
│  🔴🔴  │  🔵🔵  │  ← Class 0 very similar to each other, different from Class 1
├───────┼───────┤
│  🔵🔵  │  🔴🔴  │  ← Class 1 very similar to each other, different from Class 0
└───────┴───────┘
```

---

### 5. Decision Boundary Visualization

**What it does**: Shows the actual boundary between classes in 2D space.

**Best for**:
- Understanding where the model makes mistakes
- Identifying hard examples (near boundary)
- Seeing confidence regions

**How to run**:
```bash
python scripts/visualize_decision_boundary.py \
    --checkpoint outputs/contrastive/best_model.pth \
    --data_dir data/ \
    --encoder resnet50 \
    --reduction_method tsne \
    --output_dir outputs/decision_boundary
```

**What you get**:

**Plot 1: Decision Boundary**
```
Shows:
- Background color indicates predicted class region
- Black line = decision boundary
- Points = actual samples
- Yellow X = misclassified samples

Interpretation:
- Sharp boundary = confident separation
- Wiggly boundary = uncertain separation
- Samples far from boundary = easy examples
- Samples near boundary = hard examples
```

**Plot 2: Confidence Map**
```
Shows model confidence:
🟢 Green regions  = High confidence (>80%)
🟡 Yellow regions = Medium confidence (50-80%)
🔴 Red regions    = Low confidence (<50%)

Look for:
- Green everywhere except near boundary ✅ Good!
- Lots of red/yellow areas ❌ Model is uncertain
```

**Plot 3: Density Analysis**
```
Shows where samples concentrate:
- Bright areas = Many samples
- Dark areas = Few samples

Check:
- Are classes concentrated in different regions? ✅
- Is there overlap in high-density areas? ❌
```

**Plot 4: Margin Analysis**
```
Shows distance from decision boundary:
- Samples far from boundary = Confidently classified
- Samples near boundary = Hard examples (may need attention)

Red circles = Hardest 20% of examples
→ These are the most confusing samples for the model
```

---

## Quantitative Metrics

The scripts also compute numerical metrics to evaluate separation quality:

### Silhouette Score
- **Range**: -1 to +1
- **Higher is better**
- **Meaning**: Measures how well samples cluster with their own class vs other classes
- **Good**: > 0.5 (well-separated)
- **Bad**: < 0.2 (overlapping classes)

### Davies-Bouldin Index
- **Range**: 0 to ∞
- **Lower is better**
- **Meaning**: Ratio of within-cluster to between-cluster distances
- **Good**: < 1.0 (compact, separated clusters)
- **Bad**: > 2.0 (loose, overlapping clusters)

### Calinski-Harabasz Index
- **Range**: 0 to ∞
- **Higher is better**
- **Meaning**: Ratio of between-cluster to within-cluster variance
- **Good**: > 100 (well-defined clusters)
- **Bad**: < 50 (poorly defined clusters)

### Separation Ratio
- **Custom metric**: Inter-class distance / Intra-class distance
- **Higher is better**
- **Meaning**: How much farther apart different classes are compared to samples within same class
- **Good**: > 2.0 (classes are 2x farther apart than within-class spread)
- **Bad**: < 1.0 (classes overlap)

---

## Complete Workflow Example

### Step 1: Train your model
```bash
python scripts/train_contrastive.py \
    --data_dir data/ \
    --encoder resnet50 \
    --batch_size 64 \
    --epochs 200 \
    --output_dir outputs/contrastive
```

### Step 2: Visualize embeddings (all methods)
```bash
python scripts/visualize_embeddings.py \
    --checkpoint outputs/contrastive/best_model.pth \
    --data_dir data/ \
    --encoder resnet50 \
    --methods tsne umap pca similarity \
    --output_dir outputs/visualizations
```

This creates:
- `tsne_2d.png` - t-SNE plot
- `umap_2d.png` - UMAP plot
- `pca_2d.png` - PCA plot
- `similarity_matrix.png` - Similarity heatmap
- `separation_metrics.txt` - Numerical metrics

### Step 3: Visualize decision boundaries
```bash
python scripts/visualize_decision_boundary.py \
    --checkpoint outputs/contrastive/best_model.pth \
    --data_dir data/ \
    --encoder resnet50 \
    --reduction_method tsne \
    --output_dir outputs/decision_boundary
```

This creates:
- `decision_boundary.png` - Boundary with misclassified points
- `density_boundary.png` - Sample density per class
- `margin_analysis.png` - Hard examples near boundary

### Step 4: Interpret results

**Scenario 1: Good separation** ✅
```
Metrics:
- Silhouette: 0.65
- Davies-Bouldin: 0.8
- Separation Ratio: 3.2

Visuals:
- t-SNE shows distinct clusters
- Few misclassified points
- High confidence everywhere except boundary

→ Contrastive learning worked well!
→ Model learned discriminative features
→ Ready for downstream tasks
```

**Scenario 2: Poor separation** ❌
```
Metrics:
- Silhouette: 0.15
- Davies-Bouldin: 2.5
- Separation Ratio: 0.8

Visuals:
- t-SNE shows overlapping clusters
- Many misclassified points
- Low confidence regions

→ Model hasn't learned to separate classes
→ Try: longer training, stronger augmentation, larger batch size
```

**Scenario 3: Partial success** ⚠️
```
Metrics:
- Silhouette: 0.40
- Several misclassified points concentrated in one region

Visuals:
- Most samples well-separated
- One subgroup overlapping between classes

→ Some wood types are very similar
→ May need: more diverse augmentation, or accept that these are inherently similar
→ Consider collecting more data for confusing classes
```

---

## 3D Visualizations

For even better understanding, create 3D plots:

```bash
python scripts/visualize_embeddings.py \
    --checkpoint outputs/contrastive/best_model.pth \
    --data_dir data/ \
    --methods 3d \
    --output_dir outputs/visualizations
```

3D plots let you:
- Rotate to see structure from different angles
- Better understand cluster separation
- Identify outliers more easily

---

## Comparing Before vs After Training

To see the impact of contrastive learning:

### Before training (random weights):
```bash
python scripts/visualize_embeddings.py \
    --checkpoint outputs/contrastive/checkpoint_epoch_1.pth \  # Early checkpoint
    --data_dir data/ \
    --methods tsne \
    --output_dir outputs/viz_before
```

### After training:
```bash
python scripts/visualize_embeddings.py \
    --checkpoint outputs/contrastive/best_model.pth \
    --data_dir data/ \
    --methods tsne \
    --output_dir outputs/viz_after
```

**What to expect**:
- **Before**: Random scatter, no clear clusters, classes completely mixed
- **After**: Clear clusters, classes separated, similar samples grouped together

This shows that contrastive learning actually learned meaningful representations!

---

## Tips for Better Visualizations

1. **Use enough samples**:
   - Minimum: 100 samples
   - Ideal: 500+ samples
   - Too few samples → visualizations may not be representative

2. **Try multiple methods**:
   - t-SNE and UMAP can give different views
   - PCA is fast for initial exploration
   - Use all three to get complete picture

3. **Compare across training**:
   - Save checkpoints during training
   - Visualize at epochs 10, 50, 100, 200
   - See how clusters form over time

4. **Look for**:
   - Tight clusters = Good intra-class similarity
   - Large gaps between clusters = Good inter-class separation
   - Outliers = Potential mislabeled or unique samples

5. **Encoder vs Projection**:
   ```bash
   # Encoder features (use these for downstream tasks)
   python scripts/visualize_embeddings.py ...

   # Projection features (used during contrastive training)
   python scripts/visualize_embeddings.py ... --use_projection
   ```
   Usually encoder features are better for visualization and downstream tasks.

---

## Summary

Geometric visualizations help you:

✅ **Verify** that contrastive learning is working
✅ **Diagnose** issues with training
✅ **Identify** hard examples that need attention
✅ **Understand** what the model has learned
✅ **Compare** different models or training strategies
✅ **Communicate** results to others visually

The decision boundary is NOT explicitly programmed - it **emerges** from the contrastive learning process as the model learns to push different classes apart in embedding space!
