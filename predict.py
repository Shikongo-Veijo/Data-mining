# Required Python Packages
print("Importing python packages...")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
import numpy
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from sklearn.externals import joblib
print("Done importing python packages...")



# File Paths
input_path= "/home/country_boy/Desktop/dwm/loan_dataset.csv"


#Describe the dataset in terms of mean, median, stdDev....
def dataset_statistics(dataset):
    """
    Basic statistics of the dataset
    :param : dataset .csv file
    :return: None, print the basic statistics of the dataset
    """
    print (dataset.describe())

def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf



def main():
    """
    Main function
    :return:
    """
    # Load the csv file into pandas dataframe
    dataset = pd.read_csv(input_path)
    # Get basic statistics of the loaded dataset
    dataset_statistics(dataset)

    #Separating the dataset
    training_headers = ['Dependents',  'Applicant_Income', 'Credit_History',  'Coapplicant_Income'] #'Credit_Bureau_Record','Education','Property_Area', 
    target_header = 'Loan_Status'

    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[training_headers], dataset[target_header],
                                                        train_size = 0.7)
    # Train and Test dataset size details
    print ("Train_x Shape : ", train_x.shape)
    print ("Train_y Shape : ", train_y.shape)
    print ("Test_x Shape : ", test_x.shape)
    print ("Test_y Shape : ", test_y.shape)
    print ("")


    # Create random forest classifier instance
    trained_model = random_forest_classifier(train_x, train_y)
    print ("Trained model : ", trained_model)
    predictions = trained_model.predict(test_x)
    print ("")
    for i in range(0, 17):
    	print ("Actual outcome : {} and Predicted outcome : {}".format(list(test_y)[i], predictions[i]))
    
    # load the model from dis
    filename = "trained_model.sav"
    joblib.dump(trained_model, filename)
    print ("")

    # Train and Test Accuracy
    print ("Train Accuracy : ", accuracy_score(train_y, trained_model.predict(train_x)))
    print ("Test Accuracy  : ", accuracy_score(test_y, predictions))
    
    # load the saved model from disk(For testing purposes)
    loaded_model = joblib.load(filename)
    result = loaded_model.score(test_x,test_y)
    print (">>>>>>>>>>>>>>>>>>>RESULTS<<<<<<<<<<<<<<<")
    print (result)
if __name__ == "__main__":
    main()

