import pandas as pd
import numpy as np
import random
import math
import copy

time_between = [] #List of time intervals (days) between preventive maintenance 
cost = [] #List of costs associated to each time interval

epoch = 1 #start at interval of 1 day and then increment in 1 day
cost_p = 500 #Cost of preventive maintenance
cost_c = 5000 #Cost of corrective maintenance


while epoch<366:

   
    i=1
    cost_total= 0
   
    interval=epoch
    p=0
    c=0

    days_wo_maintenance=i

    while i<366:

        erro = 1- math.exp(-days_wo_maintenance/100000) #Probability of failure of machine according to the days that have gone by since last maintenance check
        prop = round(random.uniform(0,1),10) #random number to check probability of failure

        if prop<erro:
            #print("erro: ",erro,"  prop:",prop)
            #print("MACHINE FAILED")
            cost_total = cost_total + cost_c #add to total cost the cost of a corrective intervention
            days_wo_maintenance = 1 #restore the days since the last intervention to the next iteration
            c=c+1 #number of corrective interventions in a year

        
        elif interval!=0 and math.fmod(i,interval) == 0 : #if the machine didn't fail and it is a day to do a preventive intervention
            cost_total = cost_total + cost_p
            days_wo_maintenanceo=1 #restore the days since the last intervention to the next iteration
            p=p+1 #number of preventive interventions in a year
        
        else: #if the machine didn't fail and it isn't a day to do a preventive intervention
            days_wo_maintenance= days_wo_maintenance + 1 #one more day without intervention and probability of error rises

            
        i=i+1

    print("Number of preventive interventions:",p)
    print("Time in days between preventive maintenance: ",epoch)
    print("Number of times machine failed: ",c)
    print("Total cost associated:",cost_total)
    
    time_between.append(epoch)
    cost.append(cost_total)
    
    
    epoch=epoch+1

#BAR CHART

import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(time_between, cost, color ='blue',
        width = 0.4)
 
plt.xlabel("Days Between Preventive Maintenance")
plt.ylabel("Total Cost")
plt.title("Examples of Yearly Maintenance")
plt.show()
    