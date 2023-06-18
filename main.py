import pandas as pd
import nltk
import naive_bayes as nb
import argparse

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def train_and_play(data_path='./data/spamEmailDataset.csv'):
    
    ## Read the data
    file = data_path
    samples_csv = pd.read_csv(file, encoding='latin-1')
    samples_csv = samples_csv.sample(frac=1, random_state=42)
    document_df = samples_csv

    # Split and train model

    train_df, test_df = nb.split_train_test(document_df, train_percent=0.8)

    classes = ["spam", "ham"] 
    logprior, likelihood, vocabulary = nb.train_naive_bayes(train_df, classes)

    while True:
        user_input = input("Enter a string to classify (or 'quit' to exit): ")
        
        if user_input == 'quit':
            break
        
        # Call your function2 with the string
        result = nb.predict_class(user_input, logprior, likelihood, classes, vocabulary)
        
        # Display the result
        print("Class:", result)


def export_confusion_matrix(data_path='./data/spamEmailDataset.csv'):
    print("Executing export_confusion_matrix...")
    file = data_path
    samples_csv = pd.read_csv(file, encoding='latin-1')
    samples_csv = samples_csv.sample(frac=1, random_state=42)
    document_df = samples_csv

    # Split and train model

    train_df, test_df = nb.split_train_test(document_df, train_percent=0.8)

    classes = ["spam", "ham"] 
    logprior, likelihood, vocabulary = nb.train_naive_bayes(train_df, classes)
    
    percent = 10
    n = int(len(test_df) * (percent / 100))
    test_df = test_df.sample(n=n)  
    predicted_classes = []
    
    # Calculate confusion matrix
    # Iterate over the test data and predict the class for each document

    # Create the confusion matrix
    for _, row in test_df.iterrows():
        testdoc = row['message']
        predicted_class = nb.predict_class(testdoc, logprior, likelihood, classes, vocabulary)

        predicted_classes.append(predicted_class)

    # Create a DataFrame from the confusion matrix
    confusion_mat = nb.confusion_matrix(test_df['tag'], predicted_classes, labels=classes)
    confusion_df = pd.DataFrame(confusion_mat, index=classes, columns=classes)
    
    precision = nb.get_precision(confusion_df)
    accuracy = nb.get_accuracy(confusion_df)
    recall = nb.get_recall(confusion_df)

    print(f"Precision for this model is {precision:.2f}\nAccuracy is {accuracy:.2f} ]\nRecall is {recall:.2f}")
    # Save the output to a text file
    with open('results.txt', 'w') as file:
        file.write("Confusion Matrix:\n")
        file.write(str(confusion_df) + "\n")
        file.write("Precision: {}\n".format(precision))
        file.write("Accuracy: {}\n".format(accuracy))
        file.write("Recall: {}\n".format(recall))

def main():
    parser = argparse.ArgumentParser(description='Script Description')

    parser.add_argument('--train_and_play', action='store_true', help='Execute play_function')
    parser.add_argument('--export_confusion_matrix', action='store_true', help='Execute export_confusion_matrix')
    args = parser.parse_args()

    # Check the provided arguments and execute the corresponding functions
    if args.train_and_play:
        train_and_play()
    if args.export_confusion_matrix:
        export_confusion_matrix()
    else:
        print("No valid argument provided.")


if __name__ == '__main__':
    main()




