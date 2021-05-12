import pandas as pd
import numpy as np
import seaborn as sns

sns.set_style('whitegrid')


prod_avg = 100
prod_std_dev = np.sqrt(30)

cli_avg = 90
cli_std_dev = 5

num_reps = 10000
#num_simulations = 1000


prod_arr = np.random.normal(prod_avg, prod_std_dev, num_reps).round(2)

cli_arr = np.random.normal(cli_avg, cli_std_dev, num_reps).round(2)

#print(prod_arr.shape)
#print(cli_arr.shape)

demand_met = 0
demand_not_met = 0

#print(len(prod_arr))

for i in range(len(prod_arr)):
    if(prod_arr[i] >= cli_arr[i]):
        #Demand met
        demand_met +=  1
    else:
        #Demand not met
        demand_not_met += 1


print("In 100 occurances there were " + str(demand_met) +
      " satisfied and " + str(demand_not_met) + " not satisfied !")
print("Percentage of clients satisfied :" + str(demand_met*100/num_reps))