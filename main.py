import pandas as pd
import nltk
import naive_bayes as nb

nltk.download('punkt')

def main():
    file = ('./data/spamEmailDataset.csv')
    samples_csv = pd.read_csv(file, encoding='latin-1')
    samples_csv = samples_csv.sample(frac=1, random_state=42)
    samples_csv.head()

    percent = 20
    n = int(len(samples_csv) * (percent / 100))
    samples_csv = samples_csv.sample(n=n, random_state=42)
    print(len(samples_csv))

    document_df = samples_csv

    train_df, test_df = nb.split_train_test(document_df, train_percent=0.8)

    classes = ["spam", "ham"] 
    logprior, likelihood, vocabulary = nb.train_naive_bayes(train_df, classes)
    predicted_classes = []
    
    # Iterate over the test data and predict the class for each document
    for _, row in test_df.iterrows():
        testdoc = row['message']
        predicted_class = nb.predict_class(testdoc, logprior, likelihood, classes, vocabulary)
        predicted_classes.append(predicted_class)

    # Create the confusion matrix
    confusion_mat = nb.confusion_matrix(test_df['tag'], predicted_classes, labels=classes)

    # Create a DataFrame from the confusion matrix
    confusion_df = pd.DataFrame(confusion_mat, index=classes, columns=classes)
    precision = nb.get_precision(confusion_df)
    accuracy = nb.get_accuracy(confusion_df)
    print(precision, accuracy)

if __name__ == '__main__':
    main()




