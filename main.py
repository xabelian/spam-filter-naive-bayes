import pandas as pd
import nltk
import naive_bayes as nb

nltk.download('punkt')

def main():

    ## Read the data
    file = ('./data/spamEmailDataset.csv')
    samples_csv = pd.read_csv(file, encoding='latin-1')
    samples_csv = samples_csv.sample(frac=1, random_state=42)
    
    

    document_df = samples_csv

    # Split and train model

    train_df, test_df = nb.split_train_test(document_df, train_percent=0.8)

    #count ham and spam amount in train_df

    
    classes = ["spam", "ham"] 
    logprior, likelihood, vocabulary = nb.train_naive_bayes(train_df, classes, show_steps=False)
    
    spammy_string = "Get the cheapest of the cheapeast domains, hosting, and email accounts at www.cheapo.com"
    hammy_string = "Hello, how are you doing today? Will you and Javier come tomorrow? I will be at home around noon"

    spammy_string_2 = "You have won a free ticket to the USA this summer! Click here to claim your prize!"
    hammy_string_2 = "Good morning sir. Due to lack of time I won't be able to finish the project today. I will send it to you tomorrow."
    
    predicted_class_1 = nb.predict_class(spammy_string, logprior, likelihood, classes, vocabulary)
    print(f"The predicted class for the spammy string is {predicted_class_1}")
    predicted_class_2 = nb.predict_class(hammy_string, logprior, likelihood, classes, vocabulary)
    print(f"The predicted class for the hammy string is {predicted_class_2}")
    predicted_class_3 = nb.predict_class(spammy_string_2, logprior, likelihood, classes, vocabulary)
    print(f"The predicted class for the spammy string is {predicted_class_3}")
    predicted_class_4 = nb.predict_class(hammy_string_2, logprior, likelihood, classes, vocabulary)
    print(f"The predicted class for the hammy string is {predicted_class_4}")
    
    #taking only this percentage for testing now 
    percent = 10
    n = int(len(test_df) * (percent / 100))
    test_df = test_df.sample(n=n, random_state=42)

    predicted_classes = []
    
    # Calculate confusion matrix
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
    recall = nb.get_recall(confusion_df)

    print(f"Precision for this model is {precision:.2f}\nAccuracy is {accuracy:.2f} ]\nRecall is {recall:.2f}")

if __name__ == '__main__':
    main()




