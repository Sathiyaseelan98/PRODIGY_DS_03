# PRODIGY_DS_03
Analyze and Visualize sentimental pattern
<div style="background-color: Pink; padding: 15px; border-radius: 10px;">
    <p style="color: Black; font-size: 15px; font-weight: bold;"> 
        Import Libraries
    </p>
</div>
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, init
import plotly.express as px

import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from tqdm.notebook import tqdm
from collections import Counter
from wordcloud import WordCloud

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

import warnings
warnings.filterwarnings('ignore')
<div style="background-color: Pink; padding: 15px; border-radius: 10px;">
    <p style="color: black; font-size: 15px; font-weight: bold;"> 
        DATA UNDERSTANDING
    </p>
</div>
df = pd.read_csv("/kaggle/input/social-media-sentiments-analysis-dataset/sentimentdataset.csv")
df.head()
def null_count():
    return pd.DataFrame({'features': df.columns,
                'dtypes': df.dtypes.values,
                'NaN count': df.isnull().sum().values,
                'NaN percentage': df.isnull().sum().values/df.shape[0]}).style.background_gradient(cmap='Set3',low=0.1,high=0.01)
null_count()
df.duplicated().sum()
df.columns
for column in df.columns:
    num_distinct_values = len(df[column].unique())
    print(f"{column}: {num_distinct_values} distinct values")
<div style="background-color: Pink; padding: 15px; border-radius: 10px;">
    <p style="color: Black; font-size: 15px; font-weight: bold;"> 
        DATA PRE-PROCESSING
    </p>
</div>
<div style="background-color: Royalblue; padding: 15px; border-radius: 10px;">
    <p style="color: white; font-size: 15px; font-weight: bold;"> 
        Remove Columns
    </p>
</div>
df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'Hashtags','Day', 'Hour','Sentiment'])
<div style="background-color: Royalblue; padding: 15px; border-radius: 10px;">
    <p style="color: white; font-size: 15px; font-weight: bold;"> 
        Social Media (Platform)
    </p>
</div>
df['Platform'].value_counts()
df['Platform'] = df['Platform'].str.strip()
<div style="background-color: Royalblue; padding: 15px; border-radius: 10px;">
    <p style="color: white; font-size: 15px; font-weight: bold;"> 
        Country
    </p>
</div>
df['Country'].value_counts()
df['Country'] = df['Country'].str.strip()
<div style="background-color: Royalblue; padding: 15px; border-radius: 10px;">
    <p style="color: white; font-size: 15px; font-weight: bold;"> 
        Timestamp
    </p>
</div>
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

df['Day_of_Week'] = df['Timestamp'].dt.day_name()
<div style="background-color: Royalblue; padding: 15px; border-radius: 10px;">
    <p style="color: white; font-size: 15px; font-weight: bold;"> 
        Month
    </p>
</div>
month_mapping = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
}

df['Month'] = df['Month'].map(month_mapping)

df['Month'] = df['Month'].astype('object')
<div style="background-color: Royalblue; padding: 15px; border-radius: 10px;">
    <p style="color: white; font-size: 15px; font-weight: bold;"> 
        Text 
    </p>
</div>
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)  
    text = " ".join(text.split())
    tokens = word_tokenize(text)
    
    cleaned_tokens = [stemmer.stem(token) for token in tokens if token.lower() not in stop_words]
   
    cleaned_text = ' '.join(cleaned_tokens)
    
    return cleaned_text

df["Clean_Text"] = df["Text"].apply(clean)
<div style="background-color: Royalblue; padding: 15px; border-radius: 10px;">
    <p style="color: white; font-size: 15px; font-weight: bold;"> 
        Unique Columns
    </p>
</div>
specified_columns = ['Platform','Country', 'Year','Month','Day_of_Week']

for col in specified_columns:
    total_unique_values = df[col].nunique()
    print(f'Total unique values for {col}: {total_unique_values}')

    top_values = df[col].value_counts()

    colors = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE, Fore.LIGHTBLACK_EX, Fore.LIGHTRED_EX, Fore.LIGHTGREEN_EX]

    for i, (value, count) in enumerate(top_values.items()):
        color = colors[i % len(colors)]
        print(f'{color}{value}: {count}{Fore.RESET}')

    print('\n' + '=' * 30 + '\n')
<div style="background-color: Pink; padding: 15px; border-radius: 10px;">
    <p style="color: black; font-size: 15px; font-weight: bold;"> 
        EXPLORATORY DATA ANALYSIS
    </p>
</div>
df1 = df.copy()
<div style="background-color: Royalblue; padding: 15px; border-radius: 10px;">
    <p style="color: white; font-size: 15px; font-weight: bold;"> 
        Sentiment Analysis
    </p>
</div>
analyzer = SentimentIntensityAnalyzer()

df1['Vader_Score'] = df1['Clean_Text'].apply(lambda text: analyzer.polarity_scores(text)['compound'])

df1['Sentiment'] = df1['Vader_Score'].apply(lambda score: 'positive' if score >= 0.05 else ('negative' if score <= -0.05 else 'neutral'))

print(df1[['Clean_Text', 'Vader_Score', 'Sentiment']].head())
colors = ['#DC381F', '#2B65EC', '#01F9C6']

explode = (0.1, 0, 0)  

sentiment_counts = df1.groupby("Sentiment").size()

fig, ax = plt.subplots()

wedges, texts, autotexts = ax.pie(
    x=sentiment_counts, 
    labels=sentiment_counts.index,
    autopct=lambda p: f'{p:.2f}%\n({int(p*sum(sentiment_counts)/100)})', 
    wedgeprops=dict(width=0.7),
    textprops=dict(size=10, color="black"),  
    pctdistance=0.7,
    colors=colors,
    explode=explode,
    shadow=True)

center_circle = plt.Circle((0, 0), 0.6, color='white', fc='white', linewidth=1.25)
fig.gca().add_artist(center_circle)

ax.text(0, 0, 'Sentiment\nDistribution', ha='center', va='center', fontsize=14, fontweight='bold', color='#333333')

ax.legend(sentiment_counts.index, title="Sentiment", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

ax.axis('equal')  

plt.show()
<div style="background-color: Yellow; padding: 15px; border-radius: 10px;">
    <p style="color: Black; font-size: 15px; font-weight: bold;"> 
        Year
    </p>
</div>
plt.figure(figsize=(12, 6))
sns.countplot(x='Year', hue='Sentiment', data=df1, palette='Paired')
plt.title('Relationship between Years and Sentiment')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
<div style="background-color: Yellow; padding: 15px; border-radius: 10px;">
    <p style="color: Black; font-size: 15px; font-weight: bold;"> 
        Month
    </p>
</div>
plt.figure(figsize=(12, 6))
sns.countplot(x='Month', hue='Sentiment', data=df1, palette='Paired')
plt.title('Relationship between Month and Sentiment')
plt.xlabel('Month')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
<div style="background-color: Yellow; padding: 15px; border-radius: 10px;">
    <p style="color: Black; font-size: 15px; font-weight: bold;"> 
        Day of Week
    </p>
</div>
plt.figure(figsize=(12, 6))
sns.countplot(x='Day_of_Week', hue='Sentiment', data=df1, palette='Paired')
plt.title('Relationship between Day of Week and Sentiment')
plt.xlabel('Day of Week')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
<div style="background-color: Yellow; padding: 15px; border-radius: 10px;">
    <p style="color: Black; font-size: 15px; font-weight: bold;"> 
        Platform
    </p>
</div>
plt.figure(figsize=(12, 6))
sns.countplot(x='Platform', hue='Sentiment', data=df1, palette='Paired')
plt.title('Relationship between Platform and Sentiment')
plt.xlabel('Platform')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
<div style="background-color: Yellow; padding: 15px; border-radius: 10px;">
    <p style="color: Black; font-size: 15px; font-weight: bold;"> 
        Country
    </p>
</div>
plt.figure(figsize=(12, 6))

top_10_countries = df1['Country'].value_counts().head(10).index

df_top_10_countries = df1[df1['Country'].isin(top_10_countries)]

sns.countplot(x='Country', hue='Sentiment', data=df_top_10_countries, palette='Paired')
plt.title('Relationship between Country and Sentiment (Top 10 Countries)')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
<div style="background-color: pink; padding: 15px; border-radius: 10px;">
    <p style="color: Black; font-size: 15px; font-weight: bold;"> 
        MOST COMMON WORD
    </p>
</div>
df1['temp_list'] = df1['Clean_Text'].apply(lambda x: str(x).split())
top_words = Counter([item for sublist in df1['temp_list'] for item in sublist])
top_words_df = pd.DataFrame(top_words.most_common(20), columns=['Common_words', 'count'])

top_words_df.style.background_gradient(cmap='Blues')
df1['temp_list'] = df1['Clean_Text'].apply(lambda x: str(x).split())
top_words = Counter([item for sublist in df1['temp_list'] for item in sublist])
top_words_df = pd.DataFrame(top_words.most_common(20), columns=['Common_words', 'count'])

fig = px.bar(top_words_df,
            x="count",
            y="Common_words",
            title='Common Words in Text Data',
            orientation='h',
            width=700,
            height=700,
            color='Common_words')

fig.show()
<div style="background-color: Blue; padding: 15px; border-radius: 10px;">
    <p style="color: White; font-size: 15px; font-weight: bold;"> 
        Positive Word
    </p>
</div>
top = Counter([item for sublist in df1[df1['Sentiment'] == 'positive']['temp_list'] for item in sublist])
temp_positive = pd.DataFrame(top.most_common(10), columns=['Common_words', 'count'])
temp_positive.style.background_gradient(cmap='Blues')
words = ' '.join([item for sublist in df1[df1['Sentiment'] == 'positive']['temp_list'] for item in sublist])
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(words)

plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
<div style="background-color: Green; padding: 15px; border-radius: 10px;">
    <p style="color: White; font-size: 15px; font-weight: bold;"> 
        Neutral Word
    </p>
</div>
top = Counter([item for sublist in df1[df1['Sentiment'] == 'neutral']['temp_list'] for item in sublist])
temp_positive = pd.DataFrame(top.most_common(10), columns=['Common_words', 'count'])
temp_positive.style.background_gradient(cmap='Greens')
words = ' '.join([item for sublist in df1[df1['Sentiment'] == 'neutral']['temp_list'] for item in sublist])
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(words)

plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
<div style="background-color: Red; padding: 15px; border-radius: 10px;">
    <p style="color: White; font-size: 15px; font-weight: bold;"> 
        Negative Word
    </p>
</div>
top = Counter([item for sublist in df1[df1['Sentiment'] == 'negative']['temp_list'] for item in sublist])
temp_positive = pd.DataFrame(top.most_common(10), columns=['Common_words', 'count'])
temp_positive.style.background_gradient(cmap='Reds')
words = ' '.join([item for sublist in df1[df1['Sentiment'] == 'negative']['temp_list'] for item in sublist])
wordcloud = WordCloud(width=800, height=400, background_color='Black').generate(words)

plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
<div style="background-color: Pink; padding: 15px; border-radius: 10px;">
    <p style="color: Black; font-size: 15px; font-weight: bold;"> 
        DATA PREPARATION
    </p>
</div>
df2 = df1.copy()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
<div style="background-color: Yellow; padding: 15px; border-radius: 10px;">
    <p style="color: Black; font-size: 15px; font-weight: bold;"> 
        Data Splitting
    </p>
</div>
X = df2['Clean_Text'].values
y = df2['Sentiment'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)
<div style="background-color: Yellow; padding: 15px; border-radius: 10px;">
    <p style="color: Black; font-size: 15px; font-weight: bold;"> 
        Build Model (Machine Learning) 
    </p>
</div>
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
<div style="background-color: Blue; padding: 15px; border-radius: 10px;">
    <p style="color: White; font-size: 15px; font-weight: bold;"> 
        Logistic Regression
    </p>
</div>
logistic_classifier = LogisticRegression(max_iter=50, random_state=42)
logistic_classifier.fit(X_train_tfidf, y_train)
y_pred_logistic = logistic_classifier.predict(X_test_tfidf)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
classification_rep_logistic = classification_report(y_test, y_pred_logistic)
<div style="background-color: Blue; padding: 15px; border-radius: 10px;">
    <p style="color: White; font-size: 15px; font-weight: bold;"> 
        Classification Report (Logistic Regression)
    </p>
</div>
print("Logistic Regression Results:")
print(f"Accuracy: {accuracy_logistic}")
print("Classification Report:\n", classification_rep_logistic)
<div style="background-color: Purple; padding: 15px; border-radius: 10px;">
    <p style="color: White; font-size: 15px; font-weight: bold;"> 
        Random Forest
    </p>
</div>
random_forest_classifier = RandomForestClassifier(random_state=42)
random_forest_classifier.fit(X_train_tfidf, y_train)
y_pred_rf = random_forest_classifier.predict(X_test_tfidf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
classification_rep_rf = classification_report(y_test, y_pred_rf)
<div style="background-color: Purple; padding: 15px; border-radius: 10px;">
    <p style="color: White; font-size: 15px; font-weight: bold;"> 
        Classification Report Random Forest
    </p>
</div>
print("\nRandom Forest Results:")
print(f"Accuracy: {accuracy_rf}")
print("Classification Report:\n", classification_rep_rf)
<div style="background-color: Orange; padding: 15px; border-radius: 10px;">
    <p style="color: Black; font-size: 15px; font-weight: bold;"> 
        Support Vector Machine (SVM)
    </p>
</div>
svm_classifier = SVC(random_state=42)
svm_classifier.fit(X_train_tfidf, y_train)
y_pred_svm = svm_classifier.predict(X_test_tfidf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
classification_rep_svm = classification_report(y_test, y_pred_svm)
<div style="background-color: Orange; padding: 15px; border-radius: 10px;">
    <p style="color: Black; font-size: 15px; font-weight: bold;"> 
        Classification Report SVM
    </p>
</div>
print("Support Vector Machine Results:")
print(f"Accuracy: {accuracy_svm}")
print("Classification Report:\n", classification_rep_svm)
<div style="background-color: Pink; padding: 15px; border-radius: 10px;">
    <p style="color: Black; font-size: 15px; font-weight: bold;"> 
        LEXICON BASED CLASSIFICATION
    </p>
</div>
**COMPARISON PURPOSE**
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Load your data into a DataFrame (modify this according to your actual data loading process)
df = pd.read_csv('/kaggle/input/social-media-sentiments-analysis-dataset/sentimentdataset.csv')

# Function to get sentiment scores
def get_sentiment(text):
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores['compound']

# Apply the function to the text column
df['sentiment_score'] = df['Text'].apply(get_sentiment)

# Classify sentiment based on the compound score
def classify_sentiment(score):
    if score > 0:
        return 'positive'
    elif score < 0:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['sentiment_score'].apply(classify_sentiment)

# Display the first few rows of the DataFrame
print(df[['Text', 'sentiment_score', 'sentiment']].head())

# (Optional) Evaluate the performance
# Assuming you have a column 'true_sentiment' with the actual sentiment labels
# from sklearn.metrics import classification_report
# print(classification_report(df['true_sentiment'], df['sentiment']))
# Save the entire DataFrame with sentiment analysis results to a CSV file
df.to_csv('/kaggle/working/sentiment_analysis_results.csv', index=False)

print("Results saved to sentiment_analysis_results.csv")
