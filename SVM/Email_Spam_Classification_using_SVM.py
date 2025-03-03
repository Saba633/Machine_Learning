import re
import seaborn as sns
import pandas as pd
import numpy as np
import nltk
from mailbox import mbox
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

# Ensure stopwords are downloaded
# nltk.download('stopwords')
# nltk.download('punkt')

# Function to clean email body
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    words = word_tokenize(text)  
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(words)

# Load the mbox file
m_box = mbox(r'D:\Code\Python\ML\All mail Including Spam and Trash.mbox') # <------your file will go here
emails = []

for message in m_box:
    subject = message['subject']
    
    if message.is_multipart():
        body = ''.join([part.get_payload(decode=True).decode('utf-8', errors='ignore') 
                        for part in message.walk() if part.get_content_type() == 'text/plain'])
    else:
        body = message.get_payload(decode=True)
        if isinstance(body, bytes):
            body = body.decode('utf-8', errors='ignore')

    if subject and body:
        labels = message.get('X-Gmail-Labels')
        label = 'spam' if labels and 'Spam' in labels else 'ham'
        emails.append([subject, clean_text(body), label])

# Convert to DataFrame
df = pd.DataFrame(emails, columns=['subject', 'body', 'label'])
df.dropna(inplace=True)

# Split dataset
X = df['body'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert text data into numerical vectors
vectorization = TfidfVectorizer()
X_train = vectorization.fit_transform(X_train)
X_test = vectorization.transform(X_test)

# Train SVM model
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🔹 Model Accuracy: {accuracy*100:.2f}%\n")

# Detailed classification report
print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# ------------------- PLOTTING -------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Spam vs Ham distribution (Pie Chart)
df['label'].value_counts().plot(
    kind='pie', 
    autopct='%1.0f%%', 
    colors=['#ff9999', '#66b3ff'], 
    ax=axes[0]
)
axes[0].set_title('Spam vs Ham Emails')
axes[0].set_ylabel('')

# Confusion Matrix (Heatmap)
sns.heatmap(
    conf_matrix, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    cbar=False, 
    xticklabels=['Ham', 'Spam'], 
    yticklabels=['Ham', 'Spam'], 
    ax=axes[1]
)
axes[1].set_title('Confusion Matrix')

# Show plots
plt.tight_layout()
plt.show()
