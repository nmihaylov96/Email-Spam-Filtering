# Email-Spam-Filtering
The goal is to explore and compare the effectiveness of different AI techniques for email spam filtering. Specifically, we will evaluate Naive Bayes, Support Vector Machine (SVM), and Artificial Neural Networks (ANN) in terms of their accuracy, efficiency, and ability to handle various characteristics of spam emails.

# Spam Detection Using Naive Bayes

This project implements a spam detection system using the Naive Bayes algorithm. The model classifies messages as either "spam" or "ham" (not spam) based on a dataset of SMS messages.

## Table of Contents

- Installation
- Usage
- Data
- Model Training
- Results
- Visualizations
- License
  
## Installation

To run this project, you need to have Python installed along with the necessary libraries. You can install the required libraries using pip:

pip install scikit-learn pandas matplotlib seaborn

## Usage

1. Clone the repository:
   git clone https://github.com/yourusername/spam-detection.git
   cd spam-detection

2. Place your CSV dataset (named `spam.csv`) in the root directory.

3. Open a Jupyter Notebook and run the following commands:

```python
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the CSV file into a DataFrame
spam_df = pd.read_csv('spam.csv', encoding='latin1')

# Data Preprocessing
spam_df.rename(columns={'v1': 'Category', 'v2': 'Message'}, inplace=True)
spam_df = spam_df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
spam_df['spam'] = spam_df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(spam_df.Message, spam_df.spam, test_size=0.3, random_state=42)

# Convert text data to numerical data
cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train.values)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(x_train_count, y_train)

# Evaluate the model
x_test_count = cv.transform(x_test)
accuracy = model.score(x_test_count, y_test)
print("Model Accuracy:", accuracy)

# Display the confusion matrix
y_pred = model.predict(x_test_count)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Spam'])
disp.plot()
plt.show()

# Visualize the distribution of ham/spam emails
plt.figure(figsize=(8, 6))
sns.barplot(x=['ham', 'spam'], y=spam_df['Category'].value_counts())
plt.title('Distribution of Messages by Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()
```
##Data
The dataset used in this project is a collection of SMS messages labeled as "spam" or "ham." The data is loaded from a CSV file, and unnecessary columns are removed to clean the dataset.

Model Training
The model is trained using the Multinomial Naive Bayes algorithm. The text messages are converted into numerical form using the CountVectorizer to facilitate the training process. The dataset is split into training and testing sets for evaluation purposes.

Results
The model achieves an accuracy of approximately 98.2% on the test dataset. A confusion matrix is displayed to visualize the performance of the model.

Visualizations
The project includes visualizations that depict:

The distribution of messages categorized as ham and spam.
The confusion matrix for a deeper insight into the model's predictions.
License
This project is licensed under the MIT License. See the LICENSE file for details.
