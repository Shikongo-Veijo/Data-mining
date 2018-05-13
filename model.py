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
def main():
	path = "/home/country_boy/Desktop/dwm/sampleData.csv"
	dataset = pd.read_csv(path)
	filename = "trained_model.sav"
	training_headers = ['Dependents',  'Applicant_Income', 'Credit_History',  'Coapplicant_Income'] #'Credit_Bureau_Record','Education','Property_Area', 
	target_header = 'Loan_Status'
	id =  'Loan_ID'

	    # Split dataset into train and test dataset
	train_x, test_x, train_y, test_y = train_test_split(dataset[training_headers], dataset[target_header],
	                                                        train_size = 0.0)
	print ("Train_x Shape : ", train_x.shape)
	print ("Train_y Shape : ", train_y.shape)
	print ("Test_x Shape : ", test_x.shape)
	print ("Test_y Shape : ", test_y.shape)

	loaded_model = joblib.load(filename)
	#result = loaded_model.score(test_x,test_y)
	predictions = loaded_model.predict(test_x)
	print (">>>>>>>>>>>>>>>>>>>RESULTS<<<<<<<<<<<<<<<<")
	#print (result)
	for i in range(0,5):
		#print (dataset[id][i])
		print ("Actual outcome : {} and Predicted outcome : {} ID : {}".format(list(test_y)[i], predictions[i], dataset[id][i]))
    	


if __name__ == "__main__":
    main()
