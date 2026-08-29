# Automated Research Paper Categorization

A deep learning-based **multi-label text classification system** that automatically categorizes research papers into relevant academic subject areas using their **title and abstract**.

Developed as part of **KRITI'24 at IIT Guwahati**.

---

## Overview

Research papers can belong to multiple academic domains simultaneously, making manual categorization time-consuming and difficult to scale.

This project automates the categorization process by taking the **title and abstract of a research paper** as input and predicting one or more relevant research categories.

The system uses an **LSTM-based neural network** to learn sequential patterns from research-paper text and performs multi-label classification using independent sigmoid outputs for each category.

### Key Highlights

- **51,210 training papers**
- **10,974 test papers**
- **57 research categories**
- Title + Abstract based classification
- **5,000-word vocabulary**
- **50-dimensional embeddings**
- **100-unit LSTM**
- Multi-label prediction using sigmoid activation
- **0.453 Micro-F1** on a 10% validation split

---

## Problem Statement

Given the title and abstract of a research paper, predict all applicable research categories.

Unlike multi-class classification, where each sample belongs to exactly one class, this is a **multi-label classification problem** because a single paper can belong to multiple research areas.

For example, a paper may simultaneously belong to:

- Artificial Intelligence (`cs.AI`)
- Machine Learning (`cs.LG`)
- Computation and Language (`cs.CL`)

The model therefore predicts an independent probability for every category.

---

## Dataset

The dataset contains research papers represented using their:

- `Id`
- `Title`
- `Abstract`
- `Categories`

### Dataset Size

| Dataset | Samples |
|---|---:|
| Training | 51,210 |
| Validation | 5,121 |
| Test | 10,974 |
| Categories | 57 |

The validation set consists of **10% of the training data**.

### Research Categories

The dataset contains 57 research-subject categories spanning areas such as:

#### Computer Science
- `cs.AI` — Artificial Intelligence
- `cs.CL` — Computation and Language
- `cs.CV` — Computer Vision
- `cs.DB` — Databases
- `cs.IR` — Information Retrieval
- `cs.LG` — Machine Learning
- `cs.RO` — Robotics
- `cs.SE` — Software Engineering
- `cs.CR` — Cryptography and Security

#### Mathematics
Categories include areas such as:
- `math.CO`
- `math.PR`
- `math.ST`
- `math.IT`
- `math.NT`

#### Statistics
- `stat.ML`
- `stat.ME`
- `stat.TH`
- `stat.AP`
- `stat.CO`

#### Other Domains
The remaining categories cover:
- Economics
- Electrical Engineering and Systems Science
- Quantitative Biology
- Quantitative Finance

---

## Approach

The complete pipeline is:

```text
Research Paper
      |
      v
Title + Abstract
      |
      v
Text Tokenization
      |
      v
Integer Sequences
      |
      v
Sequence Padding
      |
      v
Embedding Layer
      |
      v
Spatial Dropout
      |
      v
LSTM
      |
      v
Dense + Sigmoid
      |
      v
Category Probabilities
      |
      v
0.5 Threshold
      |
      v
Multi-label Predictions
      |
      v
Submission File
