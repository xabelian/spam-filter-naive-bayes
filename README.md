
# Naive Bayes Classifier for Spam Filtering

## Overview

This project implements a Naive Bayes Classifier for text classification using mostly pure Python without any ML libs. The classifier is designed to predict the class (e.g., spam or ham) of a given document.

## Usage

1.  **Data Preparation:**
    
    -   Ensure your data is in a pandas DataFrame format.
    -   The DataFrame should have a "tag" column indicating the class of each document and a "message" column containing the text of the document.
    
   ```
    train_data, test_data = split_train_test(data)
  ```
2.  **Training and Testing Split:**
    
    -   Use the `split_train_test` function to split your data into training and testing sets.
    
  ```
train_data, test_data = split_train_test(data)
  ```
    
3.  **Training the Naive Bayes Classifier:**
    
    -   Train the classifier using the training data.
    
```
    classes = ["spam", "ham"]  # Customize classes as needed
    logprior, loglikelihood, vocabulary = train_naive_bayes(train_data, classes)
```
    
4.  **Prediction:**
    
    -   Use the trained classifier to predict the class of a document.
    
```
    
    document_to_predict = "This is a test message."
    predicted_class = predict_class(document_to_predict, logprior, loglikelihood, classes, vocabulary)
```
    
5.  **Evaluation:**
    
    -   Evaluate the performance of the classifier using metrics such as accuracy, precision, and recall.
    
```
    
    # Example: Confusion Matrix
    actual_classes = test_data["tag"].tolist()
    predicted_classes = [predict_class(doc, logprior, loglikelihood, classes, vocabulary) for doc in test_data["message"]]
    confusion_matrix_df = confusion_matrix(actual_classes, predicted_classes, labels=classes)
    
    # Example: Accuracy
    accuracy = get_accuracy(confusion_matrix_df) 
   ``` 

## Requirements

-   Python 3.x
-   pandas
-   numpy
-   nltk
