# Importing python required packages
print("Importing python packages...")
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import export_graphviz
from sklearn import tree
# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt

print("Done importing python packages...")

# Function importing Dataset
def importData():
	#data path
	input_path= "/home/country_boy/Desktop/dwm/loan_dataset.csv"
	dataset = pd.read_csv(input_path)

	#Printing the dataset shape
	print ("Dataset Length: ", len(dataset))
	print ("Dataset Shape: ", dataset.shape)
	# Printing the dataset obseravtions
	#print ("Dataset: ",dataset.head())

	return dataset

def splitDataset(dataset, train_percentage, feature_headers, 
    				target_header):
 
    # Spliting the dataset into train and test
    train_x, test_x, train_y, test_y = train_test_split(dataset, train_percentage, 
    					[training_headers], [target_header])
     
    return train_x, test_x, train_x, test_y

# Function to perform training with giniIndex.
def train_Using_Gini(train_x, test_x, train_y):
 
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini",max_depth=3, 
    	min_samples_leaf=5)
 
    # Performing training
    clf_gini.fit(train_x, train_y)
    return clf_gini


 #Function to perform training with entropy.
def train_Using_Entropy(train_x, test_x, train_y):
 
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
            criterion = "entropy",max_depth = 5, min_samples_leaf = 10)
 
    # Performing training
    clf_entropy.fit(train_x, train_y)
    return clf_entropy

# Function to make predictions
def prediction(test_x, clf_object):
 
    # Predicton on test with giniIndex
    pred_y = clf_object.predict(test_x)
    print("Predicted values:")
    print(pred_y)
    return pred_y

# Function to calculate accuracy
def cal_accuracy(test_y, pred_y):
     
    print("Confusion Matrix: ",
        confusion_matrix(test_y, pred_y))
     
    print ("Accuracy : ",
    accuracy_score(test_y,pred_y)*100)
     
    print("Report : ",
    classification_report(test_y, pred_y))
# Driver code
def main():
     
    # Building Phase
    data = importData()
    #Separating the dataset
    '''training_headers = ['Dependents',  'Applicant_Income', 
    						'Credit_Bureau_Record','Education',
    						'Property_Area','Credit_History',  'Coapplicant_Income'] 
    '''
    training_headers = ['Dependents',  'Applicant_Income', 'Credit_History',  'Coapplicant_Income']
    target_header = 'Loan_Status'

    train_x, test_x, train_y, test_y = train_test_split(data[training_headers], data[target_header],
                                                        train_size = 0.7)

    clf_gini = train_Using_Gini(train_x, test_x, train_y)
    clf_entropy = train_Using_Entropy(train_x, test_x, train_y)
     
    # Operational Phase
    print("Results Using Gini Index:")
     
    # Prediction using gini
    pred_gini_y = prediction(test_x, clf_gini)
    cal_accuracy(test_y, pred_gini_y)
     
    print("Results Using Entropy:")
    # Prediction using entropy
    pred_entropy_y = prediction(test_x, clf_entropy)
    cal_accuracy(test_y, pred_entropy_y)
     
    '''with open("clf_entropy.txt", "w") as f:
    	f = tree.export_graphviz(clf_entropy , out_file=f)
    '''
    
    %matplotlib inline

    # Set the style
	plt.style.use('fivethirtyeight')

	# list of x locations for plotting
	test_x = list(range(len(importances))

	# Make a bar chart
	plt.bar(test_x, importances, orientation = 'vertical')

	# Tick labels for x axis
	plt.xticks(x_values, feature_list, rotation='vertical')

	# Axis labels and title
	plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# Calling main function
if __name__=="__main__":
    main()