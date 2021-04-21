
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
clients_coord = clients_df[['XX','YY']]


c_coord_arr = clients_coord.to_numpy()
 

#print(c_coord_arr)

#print(clients_coord[0,1])

#df.loc[df[‘column name’]condition]

### Array
### 1st position is the store
### 2nd position is the dimension ---- 0 - XX and 1 - YY
print(c_coord_arr[1, ])
print(c_coord_arr[0,])
distance = np.linalg.norm(c_coord_arr[1,] - c_coord_arr[0,])

print(distance)
#16 vetores de 20 posições















