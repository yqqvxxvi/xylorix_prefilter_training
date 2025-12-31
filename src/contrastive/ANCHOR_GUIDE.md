# Understanding Anchors in Contrastive Learning

## Table of Contents
1. [What are Anchors?](#what-are-anchors)
2. [Anchor, Positive, and Negative Pairs](#anchor-positive-and-negative-pairs)
3. [How to Set Anchors](#how-to-set-anchors)
4. [Visual Examples](#visual-examples)
5. [Implementation Details](#implementation-details)
6. [Best Practices](#best-practices)

---

## What are Anchors?

In contrastive learning, an **anchor** is simply a reference sample that we use to compare against other samples. Think of it as asking the question: "Which other samples are similar to THIS one?"

**Key Concept**: The anchor is not a special data point - it's just the perspective from which we're measuring similarity.

### In Self-Supervised Learning (SimCLR):
- **Anchor**: An augmented view of an image
- **Positive**: A different augmented view of the SAME image
- **Negative**: Augmented views of DIFFERENT images

---

## Anchor, Positive, and Negative Pairs

### The Core Idea

```
Original Image: [Photo of Oak Wood]
       |
       ├─> Augmentation 1 (crop + blur + color jitter) → VIEW 1 (Anchor)
       └─> Augmentation 2 (different crop + blur + color jitter) → VIEW 2 (Positive)

Other Image: [Photo of Pine Wood]
       └─> Augmentation → VIEW 3 (Negative)
```

**The Goal**: Make the anchor and positive similar in embedding space, while pushing the anchor and negative apart.

### Mathematical Formulation

For a batch of images, we create pairs:
- For each image `i`, we create two augmented views: `view1_i` and `view2_i`
- `(view1_i, view2_i)` form a **positive pair**
- `(view1_i, view1_j)` where `i ≠ j` are **negative pairs**
- `(view1_i, view2_j)` where `i ≠ j` are also **negative pairs**

### Why This Works

The model learns to:
1. **Recognize invariances**: Different augmentations of the same image should have similar embeddings (handles variations in viewpoint, lighting, etc.)
2. **Distinguish between different samples**: Different images should have different embeddings

---

## How to Set Anchors

### In Our Implementation

**The good news**: You don't manually select anchors! The system automatically treats each sample as an anchor.

Here's how it works:

### 1. **Data Preparation** (Automatic)

```python
from src.contrastive import get_contrastive_augmentation, ContrastiveTransformations

# Create augmentation pipeline
base_transform = get_contrastive_augmentation(
    image_size=224,
    grayscale=False,
    strength='strong'  # Options: 'weak', 'medium', 'strong'
)

# This wrapper applies augmentation twice to create two views
contrastive_transform = ContrastiveTransformations(base_transform, n_views=2)
```

### 2. **What Happens During Training**

```python
# For each image in batch:
original_image = load_image("wood_sample_001.jpg")

# Apply augmentation twice → creates two different views
view1 = contrastive_transform(original_image)  # First augmentation
view2 = contrastive_transform(original_image)  # Second augmentation (DIFFERENT)

# view1 is treated as the anchor
# view2 is the positive pair
# All other images' views in the batch are negatives
```

### 3. **Anchor Selection in the Loss Function**

The NT-Xent loss automatically handles anchor selection:

```python
# Batch of 4 images creates 8 views total (2 per image)
# Views: [view1_img0, view1_img1, view1_img2, view1_img3,
#         view2_img0, view2_img1, view2_img2, view2_img3]

# For view1_img0 (anchor):
#   - Positive: view2_img0 (same image, different augmentation)
#   - Negatives: all other 6 views (different images)

# For view2_img0 (now as anchor):
#   - Positive: view1_img0
#   - Negatives: all other 6 views
```

---

## Visual Examples

### Example 1: Oak Wood Classification

```
Batch = [oak1.jpg, pine1.jpg, oak2.jpg, mahogany1.jpg]

After augmentation (2 views each):
Anchors (View 1):         Positives (View 2):       Negatives:
┌─────────────┐          ┌─────────────┐          ┌─────────────┐
│ oak1_crop1  │  ←pair→  │ oak1_crop2  │          │ pine1_*     │
└─────────────┘          └─────────────┘          │ oak2_*      │
                                                  │ mahogany1_* │
                                                  └─────────────┘
```

### Example 2: What the Model Learns

Before training:
```
Embedding Space (random):
oak1_v1: [0.2, -0.5, 0.8, ...]
oak1_v2: [-0.7, 0.3, -0.1, ...]  ← Far apart! (BAD)
pine1_v1: [0.3, -0.4, 0.7, ...]  ← Close to oak! (BAD)
```

After training:
```
Embedding Space (learned):
oak1_v1: [0.8, 0.3, -0.2, ...]
oak1_v2: [0.82, 0.28, -0.18, ...]  ← Very close! (GOOD)
pine1_v1: [-0.5, -0.7, 0.9, ...]   ← Far from oak! (GOOD)
```

---

## Implementation Details

### Augmentation Strength and Anchor Quality

The strength of augmentation affects how "different" the positive pairs look:

#### Weak Augmentation (strength='weak')
```python
# Smaller crops (70-100% of original)
# Mild color jitter (0.4)
# Less blur (30% probability)

USE WHEN:
- Dataset is small (< 1000 images)
- Images are very similar already
- Fine-tuning from pretrained model
```

#### Strong Augmentation (strength='strong') - **RECOMMENDED**
```python
# Aggressive crops (8-100% of original)
# Strong color jitter (0.8)
# More blur (50% probability)

USE WHEN:
- Large dataset (> 5000 images)
- Self-supervised pre-training from scratch
- Need robust features
```

### Batch Size Matters!

Larger batch size = More negative samples per anchor = Better learning

```python
Batch Size    | Negatives per Anchor | Recommended For
------------- | -------------------- | ---------------
32            | 62                   | Small GPU (4-6GB)
64            | 126                  | Medium GPU (8-12GB)
128           | 254                  | Large GPU (16-24GB)
256+          | 510+                 | Multi-GPU / TPU
```

**Trade-off**: If GPU memory is limited, use smaller batch size with gradient accumulation.

---

## Best Practices

### 1. **Augmentation Design**

For wood texture classification:

```python
# GOOD: Augmentations that preserve wood identity
- Random crops (wood texture is similar across regions)
- Color jitter (lighting variations)
- Blur (camera focus variations)
- Rotation/Flip (orientation doesn't matter)

# AVOID: Augmentations that change wood type
- Don't use augmentations that fundamentally alter grain patterns
- Be careful with extreme color changes (may look like different wood)
```

### 2. **Temperature Parameter**

The temperature τ in NT-Xent loss controls focus on hard negatives:

```python
# Lower temperature (0.1 - 0.3): Focus on hardest negatives
criterion = NTXentLoss(temperature=0.1)  # Aggressive

# Medium temperature (0.4 - 0.6): Balanced
criterion = NTXentLoss(temperature=0.5)  # RECOMMENDED

# Higher temperature (0.7 - 1.0): Softer, all negatives matter
criterion = NTXentLoss(temperature=0.8)  # Gentle
```

### 3. **Training Duration**

Contrastive learning needs MORE epochs than supervised learning:

```python
# Supervised learning: 50-100 epochs
# Contrastive learning: 100-500 epochs (depending on dataset size)

# For wood classification with ~1000 images:
--epochs 200  # Good starting point
```

### 4. **Evaluation Strategy**

After pre-training, evaluate using **linear probe**:

```python
# 1. Freeze the encoder (trained with contrastive learning)
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False

# 2. Train only a linear classifier on top
classifier = nn.Linear(encoder.encoder_dim, num_classes)

# 3. This tells you how good the learned features are!
```

---

## Common Questions

### Q1: Do I need to manually choose which samples are anchors?
**A**: No! Every sample automatically becomes an anchor. The loss function handles this.

### Q2: Can I use labels in contrastive learning?
**A**: For self-supervised learning (SimCLR), labels are NOT used during training. However, you can use `SupConLoss` if you want to incorporate label information (supervised contrastive learning).

### Q3: How many augmentations should I use per image?
**A**: Most methods use 2 views (SimCLR default). Using more views can help but requires more computation.

### Q4: What if two different wood types look very similar?
**A**: This is actually OKAY! The model will learn that they're different through the accumulated signal across many samples. Some confusion is expected and natural.

### Q5: Should I use ImageNet pretrained weights?
**A**: For self-supervised learning, typically start from scratch (no pretraining). The whole point is to learn representations from YOUR data.

---

## Summary

**Setting anchors is automatic!** You don't need to manually select them. Instead, focus on:

1. ✅ Choosing appropriate augmentations for your data
2. ✅ Setting the right batch size (as large as GPU allows)
3. ✅ Tuning temperature parameter
4. ✅ Training for enough epochs
5. ✅ Evaluating learned features properly

The implementation handles the anchor-positive-negative pairing automatically. Your job is to set up the augmentation pipeline and training parameters correctly.
