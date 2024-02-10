import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sub_format=pd.read_csv("sample_submission.csv")
# Combine title and abstract
train_df['text'] = train_df['Title'] + ' ' + train_df['Abstract']
test_df['text'] = test_df['Title'] + ' ' + test_df['Abstract']

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_df['text'])

train_sequences = tokenizer.texts_to_sequences(train_df['text'])
test_sequences = tokenizer.texts_to_sequences(test_df['text'])

# Padding sequences
max_length = max([len(seq) for seq in train_sequences])
train_padded = pad_sequences(train_sequences, maxlen=max_length)
test_padded = pad_sequences(test_sequences, maxlen=max_length)

# Convert categories to one-hot encoding
# encoder = LabelBinarizer()
# train_labels = encoder.fit_transform(train_df['Categories'])
mlb = MultiLabelBinarizer()
train_df['Categories'] = train_df['Categories'].apply(eval)  # Convert string representations to lists
train_labels = mlb.fit_transform(train_df['Categories'])

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=50, input_length=max_length))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
#model.add(Dense(train_labels.shape[1], activation='softmax'))
model.add(Dense(len(mlb.classes_), activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


batch_size = 64
epochs = 5

model.fit(train_padded, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.1)

predictions = model.predict(test_padded)
# Convert predictions to binary labels based on a threshold (e.g., 0.5)
binary_predictions = (predictions > 0.5).astype(int)

# Use inverse_transform to convert the binary labels back to the original class labels
predicted_classes = mlb.inverse_transform(binary_predictions)

# Convert predictions to binary labels
#predicted_labels = (predictions > 0.5).astype(int)
# Create submission DataFrame
# submission_df = pd.DataFrame(predicted_labels, columns=mlb.classes_)
# submission_df.insert(0, 'Id', test_df['Id'])
# submission_df.to_csv('submission_corrected.csv', index=False)

ids=test_df["Id"]
#m={}
# for cl in sub_format.columns:
#     m[cl]=0
final_output=pd.DataFrame(columns=sub_format.columns)

to_add = []

# Iterate over each set of predicted classes and the corresponding ID
for idx, classes in enumerate(predicted_classes):
    m = {cl: 0 for cl in sub_format.columns[1:]}  # Initialize the dictionary for current sample
    for sub_class in classes:
        if sub_class in m:  # Check if the predicted class is in the columns
            m[sub_class] = 1  # Update the dictionary to indicate the presence of the class
    row = [ids.iloc[idx]] + list(m.values())
    to_add.append(row)

    # Convert the accumulated rows into a DataFrame
final_output = pd.DataFrame(to_add, columns=sub_format.columns)

# Save the DataFrame to a CSV file
final_output.to_csv('submission_corrected11.csv', index=False)


to_add=[]
cnt=0
for classes in predicted_classes:
    m={}
    for cl in sub_format.columns[1:]:
        m[cl]=0
    for sub_class in eval(classes):
        m[sub_class]=1
    ans=[]
    ans.append(ids[cnt])
    to_add.append(ans+list(m.values()))
    cnt+=1

    for row in to_add:
    curr_len=len(final_output)
    final_output.loc[curr_len]=row

    final_output.to_csv('submission11.csv', index=False)