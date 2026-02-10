# AMR-Scout: Antimicrobial Resistance Prediction from Genomic FASTA

AMR-Scout is an **end-to-end machine learning pipeline** that predicts **antimicrobial resistance (AMR)** directly from raw bacterial genome sequences (FASTA files) using memory-efficient and scalable methods.

The project demonstrates how to transform raw DNA sequences into numerical features and train classical ML models under realistic constraints:  
**high dimensionality · extreme sparsity · class imbalance**

---

## Scientific Problem

**Antimicrobial resistance (AMR)** occurs when bacteria survive antibiotic treatment due to resistance genes or mutations present in their DNA.

**Goal**: Predict **Resistant** vs **Susceptible** phenotype

→ Treated as a **binary classification** problem using sequence patterns alone.

---

## Core Idea

Machine learning models cannot process DNA strings directly:
ATGCGTACGATCGATCG...

Instead, the pipeline transforms sequences into:

**DNA sequence → k-mers → sparse numerical vectors → linear classifier**

The model learns statistical associations between **k-mer patterns** and resistance phenotypes.

---

## Data Sources

### 1. Genome Sequences (FASTA)
- **Source:** PATRIC database  
- **Link:** https://www.patricbrc.org  
- **Content:** High-quality assembled bacterial genomes + standardized identifiers

### 2. Phenotype Labels
- **Source:** PATRIC metadata tables (CSV)  
- **Key columns:**
  - `assembly_id` – unique genome identifier
  - `Resistant Phenotype` – "Resistant" / "Susceptible"

---

## Linking Genomes and Labels

FASTA files and phenotype tables are separate → matching is required.

**Typical workflow:**

1. Extract all `assembly_id` from phenotype CSV
2. Parse FASTA headers to find matching genomes
3. Filter to keep only genomes present in both sources
4. Standardize FASTA headers to exactly match `assembly_id`
5. Attach binary label (0 = Susceptible, 1 = Resistant) to each genome

**Result:** clean **genome ↔ label** pairs

---

## Feature Engineering: k-mers

### What is a k-mer?

Substrings of length `k` sliding over the sequence.

**Example (k=3):**
Sequence: ATGCGT
k-mers:   ATG TGC GCG CGT


### Chosen Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **k** | 6 | Substring length |
| **Alphabet** | A, C, G, T | DNA only |
| **Counting** | TF-IDF weighting | Term frequency-inverse document frequency |
| **Storage** | Sparse matrix (`scipy.sparse.csr_matrix`) | Memory efficient |
| **Vocabulary size** | ~4,000–4,100 features | For k=6 |

→ Good balance between biological signal and memory/compute cost

---

## Machine Learning Pipeline

### Feature Extraction

1. FASTA files
2. Character-level 6-mer counting
3. TfidfVectorizer (sparse output)
4. csr_matrix (tens–hundreds of thousands of rows × ~4096 columns)


### Classifiers

All models are **linear**, **sparse-friendly**, and scale well to high-dimensional genomic data.

| Model | Implementation | Key Settings |
|-------|---------------|--------------|
| Logistic Regression | `sklearn.linear_model.LogisticRegression` | `solver='saga'`, `class_weight='balanced'` |
| Linear SVM | `sklearn.svm.LinearSVC` | `class_weight='balanced'` |
| SGDClassifier | `sklearn.linear_model.SGDClassifier` | `loss='hinge'`, `'log'`, or `'modified_huber'` |

All models are trained using **cost-sensitive learning** (`class_weight='balanced'`) instead of generating synthetic samples or applying oversampling techniques.

---

## Handling Class Imbalance

AMR phenotype datasets are typically **heavily imbalanced**:

- **Resistant** samples << **Susceptible** samples

### Chosen Strategies (in preferred order)

1. **Cost-sensitive training**  
   → `class_weight='balanced'` in scikit-learn estimators

2. **Prediction threshold tuning**  
   → Optimize decision threshold on validation set to maximize F1-score or recall (especially important for detecting resistant cases)

3. **Evaluation metrics focused on minority class**
   - Precision
   - Recall
   - F1-score
   - Precision-Recall AUC (PR-AUC)

   → **Avoid** plain accuracy



---

## Typical Data Characteristics

| Property | Typical Value | Notes |
|----------|--------------|-------|
| Number of samples | Few hundred – few thousand | Whole-genome AMR studies |
| Number of features | ~4,096 (for 6-mers) | Very high, but extremely sparse |
| Positive class ratio | 10–35% (Resistant) | Depends on antibiotic and species |
| Matrix density | < 1–5% non-zero entries | Justifies sparse matrix usage |

These characteristics explain the strong preference for:

- **Linear models**
- **Sparse matrix representations** (`scipy.sparse.csr_matrix`)
- **Avoiding** tree-based models (memory explosion) and deep learning (overkill + poor scaling on sparse/high-dim data without massive datasets)
