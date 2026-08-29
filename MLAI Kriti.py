import ast
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score
)


# ============================================================
# 1. LOAD DATA
# ============================================================

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
sub_format = pd.read_csv("sample_submission.csv")

print("Training samples:", len(train_df))
print("Test samples:", len(test_df))


# ============================================================
# 2. PREPARE TEXT
#    Combine Title + Abstract
# ============================================================

train_df["text"] = (
    train_df["Title"].fillna("") + " " +
    train_df["Abstract"].fillna("")
)

test_df["text"] = (
    test_df["Title"].fillna("") + " " +
    test_df["Abstract"].fillna("")
)


# ============================================================
# 3. CONVERT CATEGORY STRINGS TO LISTS
# ============================================================

train_df["Categories"] = train_df["Categories"].apply(
    ast.literal_eval
)


# ============================================================
# 4. MULTI-LABEL ENCODING
# ============================================================

mlb = MultiLabelBinarizer()

train_labels = mlb.fit_transform(
    train_df["Categories"]
)

num_classes = len(mlb.classes_)

print("Number of categories:", num_classes)
print("Label matrix shape:", train_labels.shape)


# ============================================================
# 5. TRAIN / VALIDATION SPLIT
# ============================================================

X_text = train_df["text"].values
y = train_labels

X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_text,
    y,
    test_size=0.10,
    random_state=42
)

print("Training samples after split:", len(X_train_text))
print("Validation samples:", len(X_val_text))


# ============================================================
# 6. TOKENIZATION
#    Fit tokenizer ONLY on training text
# ============================================================

MAX_WORDS = 5000

tokenizer = Tokenizer(
    num_words=MAX_WORDS
)

tokenizer.fit_on_texts(X_train_text)


# Convert text to sequences

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_val_seq = tokenizer.texts_to_sequences(X_val_text)
X_test_seq = tokenizer.texts_to_sequences(
    test_df["text"]
)


# ============================================================
# 7. PADDING
# ============================================================

max_length = max(
    len(seq) for seq in X_train_seq
)

print("Maximum sequence length:", max_length)

X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=max_length
)

X_val_pad = pad_sequences(
    X_val_seq,
    maxlen=max_length
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=max_length
)


# ============================================================
# 8. BUILD LSTM MODEL
# ============================================================

model = Sequential()

model.add(
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=50,
        input_length=max_length
    )
)

model.add(
    SpatialDropout1D(0.2)
)

model.add(
    LSTM(
        100,
        dropout=0.2,
        recurrent_dropout=0.2
    )
)

model.add(
    Dense(
        num_classes,
        activation="sigmoid"
    )
)


# ============================================================
# 9. COMPILE MODEL
# ============================================================

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()


# ============================================================
# 10. TRAIN MODEL
# ============================================================

BATCH_SIZE = 64
EPOCHS = 5

history = model.fit(
    X_train_pad,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val_pad, y_val),
    verbose=1
)


# ============================================================
# 11. VALIDATION PREDICTIONS
# ============================================================

val_predictions = model.predict(
    X_val_pad,
    verbose=1
)


# Convert probabilities to binary labels

THRESHOLD = 0.5

val_binary = (
    val_predictions >= THRESHOLD
).astype(int)


# ============================================================
# 12. EVALUATION METRICS
# ============================================================

micro_f1 = f1_score(
    y_val,
    val_binary,
    average="micro",
    zero_division=0
)

macro_f1 = f1_score(
    y_val,
    val_binary,
    average="macro",
    zero_division=0
)

micro_precision = precision_score(
    y_val,
    val_binary,
    average="micro",
    zero_division=0
)

micro_recall = recall_score(
    y_val,
    val_binary,
    average="micro",
    zero_division=0
)


# ============================================================
# 13. PRINT FINAL VALIDATION RESULTS
# ============================================================

print("\n" + "=" * 50)
print("VALIDATION RESULTS")
print("=" * 50)

print(f"Micro-F1:          {micro_f1:.4f}")
print(f"Macro-F1:          {macro_f1:.4f}")
print(f"Micro-Precision:   {micro_precision:.4f}")
print(f"Micro-Recall:      {micro_recall:.4f}")

print("=" * 50)


# ============================================================
# 14. TRAINED MODEL → TEST PREDICTIONS
# ============================================================

test_predictions = model.predict(
    X_test_pad,
    verbose=1
)


# Convert probabilities to binary labels

test_binary = (
    test_predictions >= THRESHOLD
).astype(int)


# Convert binary predictions back to category names

predicted_classes = mlb.inverse_transform(
    test_binary
)


# ============================================================
# 15. CREATE SUBMISSION FILE
# ============================================================

ids = test_df["Id"]

rows = []

for idx, classes in enumerate(predicted_classes):

    # Start every category at 0
    label_map = {
        category: 0
        for category in sub_format.columns[1:]
    }

    # Mark predicted categories as 1
    for category in classes:

        if category in label_map:
            label_map[category] = 1

    row = [
        ids.iloc[idx]
    ] + list(label_map.values())

    rows.append(row)


final_output = pd.DataFrame(
    rows,
    columns=sub_format.columns
)


# ============================================================
# 16. SAVE SUBMISSION
# ============================================================

output_file = "submission_lstm.csv"

final_output.to_csv(
    output_file,
    index=False
)

print("\nSubmission saved as:", output_file)
print("Submission shape:", final_output.shape)
print("Test predictions:", len(predicted_classes))
print("Categories:", num_classes)
