import predict
import model
import os
menu = {}
print("______________________________________")
print("                MENU                  ")
print("______________________________________")
menu['1']="Train the model." 
menu['2']="Predict from the file."
menu['3']="Predict by inputting values."
menu['4']="Exit"

while True: 
  options=menu.keys()
  for entry in options: 
    print (entry, menu[entry])
  print("______________________________________")
  selection=input("Please Select:") 
  if selection =='1': 
    print (">>>>>>>>>>>>>>>Train the model<<<<<<<<<<<<<<<<<")
    os.system('python predict.py')
  elif selection == '2': 
    print (">>>>>>>>>>>>>>>>>>>>Predict from the file.<<<<<<<<<<<<<<<<<<")
    os.system('python model.py')
  elif selection == '3':
    print (">>>>>>>>>>>>>>Predict by inputting values<<<<<<<<<<<<<<")
  elif selection == '4': 
    break
  else: 
    print ("Unknown Option Selected!")
