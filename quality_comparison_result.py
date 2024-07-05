import pandas as pd
import matplotlib.pyplot as plt

evalution = pd.read_csv('report.csv')
#print(evalution)

evalution_results = evalution.groupby(['Frame']).sum()
evalution_results.plot.bar()
plt.show()