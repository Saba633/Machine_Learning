from mailbox import mbox
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


m_box = mbox(r'D:\Code\Python\ML\All mail Including Spam and Trash.mbox') # <---------your file path
emails = []

for message in m_box:
    subject = message['subject']

    if message.is_multipart():
        body = ''.join([part.get_payload(decode=True).decode('utf-8', errors='ignore') for part in message.walk() if part.get_content_type() == 'text/plain'])
    else:
        body = message.get_payload(decode=True)

        if isinstance(body, bytes):
            body = body.decode('utf-8', errors='ignore')
    
    if subject and body:
        labels = message.get('X-Gmail-Labels')
        if labels and 'Spam' in labels:
            label = 'spam'
        else:
            label = 'spam' if 'spam' in str(subject).lower() else 'ham'
        
        emails.append([subject, body, label])

df = pd.DataFrame(emails, columns=['subject', 'body', 'label'])
df.dropna(inplace=True)

print(df['label'].value_counts())

X = df['body'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

vectorization = CountVectorizer()

X_train = vectorization.fit_transform(X_train)
X_test = vectorization.transform(X_test)

model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy {accuracy*100:.2f}%")

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
df['label'].value_counts().plot(kind='pie', autopct='%1.0f%%', colors=['#ff9999', '#66b3ff'])
plt.title('Spam vs Ham emails')

plt.subplot(1,2,2)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('SVM Trained Model')
plt.show()



'''-------------------------methods used in this porject---------------------------
    • is_multipart(): checks whether the email has multiple parts or not
    • get_payload(): is used to retrieve content of email
    • walk(): to iterate over all parts of the email
    • get_content_type(): to get the content type 
    • isinstance: checks if the object is an instance of specified class
    -------------------------------------------------------------------------------------
'''
