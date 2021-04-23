
import pandas as pd
import numpy as np
import random
import math
#import tkinter as tk

###-----Simulated Annealing

def simulated_annealing(initial_state):
    """Peforms simulated annealing to find a solution"""
    initial_temp = 90
    final_temp = .1
    alpha = 0.01

    current_temp = initial_temp

    # Start by initializing the current state with the initial state
    current_state = initial_state
    solution = current_state

    while current_temp > final_temp:
        neighbor = random.choice(get_neighbors())

        # Check if neighbor is best so far
        cost_diff = get_cost(self.current_state) - get_cost(neighbor)

        # if the new solution is better, accept it
        if cost_diff > 0:
            solution = neighbor
        # if the new solution is not better, accept it with a probability of e^(-cost/temp)
        else:
            if random.uniform(0, 1) < math.exp(-cost_diff / current_temp):
                solution = neighbor
        # decrement the temperature
        current_temp -= alpha

    return solution


def get_cost(state):
    """Calculates cost of the argument state for your solution."""

    raise NotImplementedError


def get_neighbors(state):
    """Returns neighbors of the argument state for your solution."""
    raise NotImplementedError


#def get_total_cost(routes,cli_arr,war_arr):
   #     for warehouse in range(16):
   #           for client in range(20):
  #                     if routes[warehouse,client] == 0:
  #                             break
  #                    if client == 1 :
  #                              #calculate storage to 1st store
   #                     else 
  #                              #cost =+ calculate_cost_between_stores
                        
 #       return cost 
 #   raise NotImplementedError    
#################


#####--- GUI Implementation-----###

"""
root = tk.Tk()
#def button_storage():

def click_storage(id):
        label = tk.Label(root, text="Armazem" )


w = tk.Label(root, text="Hello Tkinter!")
k = tk.Label(root, text="My name is Rodas Rolamentos")

but = tk.Button(root, text="Armazem", command=click_storage)

k.grid(row=0, column=0)
w.grid(row=1, column=0)
but.grid(row=2, column=0) """



# root.mainloop()

#############################


##----Import data from CSV----##

clients_df = pd.read_csv("clients.txt",sep=';',header=0)
warehouses_df = pd.read_csv("warehouses.txt",sep=';',header=0)

""" print("warehouses")
print(warehouses_df)
print("clients")
"""

####################

####initial state

cli_arr = clients_df[['XX' , 'YY']].to_numpy()
war_arr = warehouses_df[['XX' , 'YY']].to_numpy()
 

#print(c_coord_arr)

#print(clients_coord[0,1])

#df.loc[df[‘column name’]condition]

### Array
### 1st position is the store
### 2nd position is the dimension ---- 0 - XX and 1 - YY
print(war_arr[1, ])
print(cli_arr[0,] )
#distance = np.linalg.norm(cli_arr[1,] - cli_arr[0,])
distance = np.linalg.norm(war_arr[1, ] - cli_arr[0, ])

### In routes store 1 is 1 but in cli_arr is in position 0 

###################################################################################
routes = [[] for i in range(0,16)] 
vector_stores=list(range(0,86))
random.shuffle(vector_stores)
print(vector_stores)

j=0
for i,store in enumerate(vector_stores):

    
    print(j)
    if i%16==0 and i!=0:
        
        j=0
    routes[j].append(store)
    j=j+1

print(routes[0])
print(routes[1])
print(routes[2])
print(routes[3])
print(routes[4])
print(routes[5])
print(routes[6])
print(routes[7])
print(routes[8])
print(routes[9])
print(routes[10])
print(routes[11])
print(routes[12])
print(routes[13])
print(routes[14])
print(routes[15])
#######################################################################################################




print(distance)
#16 vetores de 20 posições

















