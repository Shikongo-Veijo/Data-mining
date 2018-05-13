import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_excel("")

fig=plt.figure() #Plots in matplotlib reside within a figure object, use plt.figure to create new figure
#Create one or more subplots using add_subplot, because you can't create blank figure
ax = fig.add_subplot(1,1,1)
#Variable
ax.hist(df['Credit_history'],bins = 4) # Here you can play with number of bins
Labels and Tit
plt.title('Loan Amount')
plt.xlabel('loan_amount')
plt.ylabel('Credit_history')
plt.show()