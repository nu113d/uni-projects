# import basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import libraries for preprocessing
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import libraries for word cloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Word2Vec
import gensim

# Libraries for NNs
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional, GRU
from sklearn.model_selection import KFold

# Libraries for metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ----------------------------------
# read raw data
raw_df=pd.read_csv('train.csv')
print(raw_df.head())

# remove id and author columns
df = raw_df.drop(['id','author'], axis=1)
# impute null values in text features with "None"
text_columns = ['title', 'text']
for col in text_columns:
        df.loc[df[col].isnull(), col] = "None"


# Define a function to clean text by removing URLs, special characters, and extra spaces
def clean_text(text):
    text = str(text).replace(r'http[\w:/\.]+', ' ')  # Remove URLs
    text = str(text).replace(r'[^\.\w\s]', ' ')  # Remove special characters
    text = str(text).replace('[^a-zA-Z]', ' ')  # Remove non-alphabetic characters
    text = str(text).replace(r'\s\s+', ' ')  # Remove extra spaces
    text = text.lower().strip()  # Convert to lowercase and strip leading/trailing spaces
    return text

# store predicted values
y = df["label"].values
# clean the text
df["text"] = df.text.apply(clean_text)
print(df.head())

# download stopwords and punkt
nltk.download('stopwords')
nltk.download('punkt')

#Convert X to format acceptable by gensim, also remove stopwords
# NOTE: X is the training data which is a list of words inside X[0]
print('Removing stopwords')
X = []
stop_words = set(nltk.corpus.stopwords.words("english"))
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
for par in df['text'].values:
    tmp = []
    sentences = nltk.sent_tokenize(par)
    for sent in sentences:
        sent = sent.lower()
        tokens = tokenizer.tokenize(sent)
        filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w) > 1]
        tmp.extend(filtered_words)
    X.append(tmp)

# WORD2VEC
print('Building word2vec')
w2v_model = gensim.models.Word2Vec(sentences=X, vector_size=100, window=5, min_count=1)
#w2v_model.wv.most_similar("google")

# Convert X (without the stopwords) back to a df to create wordcloud and n-grams
X_text = [' '.join(tokens) for tokens in X]
cleaned_df = pd.DataFrame({'text': X_text, 'label': y})
print(cleaned_df.head())

# WORDCLOUD
wordcloud = WordCloud(background_color='black', width=600, height=400)
text_cloud = wordcloud.generate(' '.join(cleaned_df['text']))
plt.figure(figsize=(20, 30))
plt.title("All News")
plt.imshow(text_cloud)
plt.axis('off')
plt.show()

# Wordcould for 'True' label
# Concatenate the 'text' data from rows where 'label' is equal to 0
true_news = ' '.join(cleaned_df[cleaned_df['label'] == 0]['text'])
wc = wordcloud.generate(true_news)
plt.figure(figsize=(20, 30))
plt.title("True News")
plt.imshow(wc)
plt.axis('off')
plt.show()

# Wordcould for 'Fake' label
# Concatenate the 'text' data from rows where 'label' is equal to 1
fake_news = ' '.join(cleaned_df[cleaned_df['label'] == 1]['text'])
wc = wordcloud.generate(fake_news)
plt.figure(figsize=(20, 30))
plt.title("Fake News")
plt.imshow(wc)
plt.axis('off')
plt.show()

# Create a Trigram for true_news 
true_bigrams = (pd.Series(nltk.ngrams(true_news.split(), 3)).value_counts())[:20]
true_bigrams.sort_values().plot.barh(color='blue', width=0.9, figsize=(12, 8))
plt.title('Top 20 Frequently Occurring True News Trigrams')
plt.ylabel('Trigram')
plt.xlabel('Number of Occurrences')
plt.show()

# Create a Trigram for fake_news 
fake_bigrams = (pd.Series(nltk.ngrams(fake_news.split(), 3)).value_counts())[:20]
fake_bigrams.sort_values().plot.barh(color='blue', width=0.9, figsize=(12, 8))
plt.title('Top 20 Frequently Occurring Fake News Trigrams')
plt.ylabel('Trigram')
plt.xlabel('Number of Occurrences')
plt.show()


# TOKENIZE
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)

# print the "total" text rows (articles)
print(len(y))
# show histogram of words per article
plt.hist([len(x) for x in X], bins=500)
plt.title('Words per article')
plt.xlabel('# words')
plt.ylabel('# articles')
plt.show()
nos = np.array([len(x) for x in X])
#print how many of them have less than 1000 words
print(len(nos[nos  < 1000]))

# 19639 out of 20800 articles have less than 1000 words.
# So we pad the ones wth less than 1000 and truncate the rest with >1000
X = pad_sequences(X, maxlen=1000)
print(X)

# Embedding layer creates one more vector for "UNKNOWN" words
# So vocab size + 1
vocab_size = len(tokenizer.word_index) + 1
vocab = tokenizer.word_index
print(vocab_size)

weight_matrix = np.zeros((vocab_size, 100))
for word, i in vocab.items():
    weight_matrix[i] = w2v_model.wv[word]

# Build NNs
def build_lstm():
    model = Sequential()
    model.add(Embedding(vocab_size, output_dim=100, weights=[weight_matrix], trainable=False))
    model.add(LSTM(20))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_bilstm():
    model = Sequential()
    model.add(Embedding(vocab_size, output_dim=100, weights=[weight_matrix], trainable=False))
    model.add(Bidirectional(LSTM(20)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_GRU():
    model = Sequential()
    model.add(Embedding(vocab_size, output_dim=100, weights=[weight_matrix], trainable=False))
    model.add(GRU(20))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Train models
def train_model(model, X_train, y_train, X_test, y_test):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    # Train the model with 0.25 validation split and 20 epochs
    history = model.fit(X_train, y_train, epochs=20, batch_size=200,validation_split=0.25)
    return model, history


kf = KFold(n_splits=3, shuffle=True, random_state=42)

results_lstm = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
results_bilstm = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
results_gru = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

preds_lstm = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}
preds_bilstm = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}
preds_gru = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # LSTM
    model_lstm = build_lstm()
    model_lstm, history_lstm = train_model(model_lstm, X_train, y_train, X_test, y_test)
    results_lstm['train_loss'].append(history_lstm.history['loss'])
    results_lstm['val_loss'].append(history_lstm.history['val_loss'])
    results_lstm['train_acc'].append(history_lstm.history['accuracy'])
    results_lstm['val_acc'].append(history_lstm.history['val_accuracy'])

    y_pred = model_lstm.predict(X_test, verbose=0).round().astype(int)
    preds_lstm['accuracy'].append(accuracy_score(y_test, y_pred))
    preds_lstm['recall'].append(recall_score(y_test, y_pred))
    preds_lstm['precision'].append(precision_score(y_test, y_pred))
    preds_lstm['f1'].append(f1_score(y_test, y_pred))


    # BiLSTM
    model_bilstm = build_bilstm()
    model_bilstm, history_bilstm = train_model(model_bilstm, X_train, y_train, X_test, y_test)
    results_bilstm['train_loss'].append(history_bilstm.history['loss'])
    results_bilstm['val_loss'].append(history_bilstm.history['val_loss'])
    results_bilstm['train_acc'].append(history_bilstm.history['accuracy'])
    results_bilstm['val_acc'].append(history_bilstm.history['val_accuracy'])

    y_pred = model_bilstm.predict(X_test, verbose=0).round().astype(int)
    preds_bilstm['accuracy'].append(accuracy_score(y_test, y_pred))
    preds_bilstm['recall'].append(recall_score(y_test, y_pred))
    preds_bilstm['precision'].append(precision_score(y_test, y_pred))
    preds_bilstm['f1'].append(f1_score(y_test, y_pred))


    # GRU
    model_gru = build_GRU()
    model_gru, history_gru = train_model(model_gru, X_train, y_train, X_test, y_test)
    results_gru['train_loss'].append(history_gru.history['loss'])
    results_gru['val_loss'].append(history_gru.history['val_loss'])
    results_gru['train_acc'].append(history_gru.history['accuracy'])
    results_gru['val_acc'].append(history_gru.history['val_accuracy'])

    y_pred = model_gru.predict(X_test, verbose=0).round().astype(int)
    preds_gru['accuracy'].append(accuracy_score(y_test, y_pred))
    preds_gru['recall'].append(recall_score(y_test, y_pred))
    preds_gru['precision'].append(precision_score(y_test, y_pred))
    preds_gru['f1'].append(f1_score(y_test, y_pred))

# plot results
def plot_train_results(results, model_name):
    epochs = range(1, 21)
    avg_train_loss = np.mean(results['train_loss'], axis=0)
    avg_val_loss = np.mean(results['val_loss'], axis=0)
    avg_train_acc = np.mean(results['train_acc'], axis=0)
    avg_val_acc = np.mean(results['val_acc'], axis=0)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, avg_train_loss, label='Training Loss')
    plt.plot(epochs, avg_val_loss, label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, avg_train_acc, label='Training Accuracy')
    plt.plot(epochs, avg_val_acc, label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# Plotting the results for each model
plot_train_results(results_lstm, 'LSTM')
plot_train_results(results_bilstm, 'BiLSTM')
plot_train_results(results_gru, 'GRU')


# plot performance
def plot_performance(pred_results, model_names):
    metrics = ['precision', 'recall', 'f1', 'accuracy']
    metric_values = {metric: [] for metric in metrics}
    
    print("Performance Metrics Mean Values:")
    for model_name in model_names:
        precision_mean = np.mean(pred_results[model_name]['precision'])
        recall_mean = np.mean(pred_results[model_name]['recall'])
        f1_mean = np.mean(pred_results[model_name]['f1'])
        accuracy_mean = np.mean(pred_results[model_name]['accuracy'])
        
        metric_values['precision'].append(precision_mean)
        metric_values['recall'].append(recall_mean)
        metric_values['f1'].append(f1_mean)
        metric_values['accuracy'].append(accuracy_mean)
        
        print(f"{model_name}:")
        print(f"  Precision: {precision_mean:.4f}")
        print(f"  Recall: {recall_mean:.4f}")
        print(f"  F1 Score: {f1_mean:.4f}")
        print(f"  Accuracy: {accuracy_mean:.4f}\n")

    x = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    rects1 = ax.bar(x - 1.5*width, metric_values['precision'], width, label='Precision')
    rects2 = ax.bar(x - 0.5*width, metric_values['recall'], width, label='Recall')
    rects3 = ax.bar(x + 0.5*width, metric_values['accuracy'], width, label='Accuracy')
    rects4 = ax.bar(x + 1.5*width, metric_values['f1'], width, label='F1 Score')

    ax.set_ylabel('Scores')
    ax.set_title('Performance Metrics by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend(loc='lower right')

    fig.tight_layout()
    plt.show()

pred_results = {
    'LSTM': preds_lstm,
    'BiLSTM': preds_bilstm,
    'GRU': preds_gru
}
model_names = ['LSTM', 'BiLSTM', 'GRU']
plot_performance(pred_results, model_names)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')