
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


def get_total_cost(routes, cli_arr, war_arr):
      for warehouse in range(16):
            for client in range(20):

                if numpy.isnan(routes[warehouse,client]) == True :
                    # Finished the end of the route or no stores in route
                    if client == 1:
                        #route is empty --  no cost
                        break
                    else :
                        #is the last store
                        #calculate cost from last shop to the warehouse
                        cost =+ get_distance( cli_arr[routes[warehouse,client-1],] ,war_arr[warehouse,])
                        break
                    
                if client == 1:
                    #first store calculate to warehouse
                    #calculate storage to 1st store
                    cost = + \
                        get_distance(
                            war_arr[warehouse, ], cli_arr[routes[warehouse, client], ])
                else:
                    cost = + \
                        get_distance(
                            cli_arr[routes[warehouse, client], ], cli_arr[routes[warehouse, client-1], ])
                    #cost =+ calculate_cost_between_stores
    #raise NotImplementedError


def get_distance(location1, location2):
    #Example -- distance = np.linalg.norm(war_arr[1, ] - cli_arr[0, ])
    distance = np.linalg.norm(location1 - location2)
    return distance 

#################


##----Import data from CSV----##

clients_df = pd.read_csv("clients.txt",sep=';',header=0)
warehouses_df = pd.read_csv("warehouses.txt",sep=';',header=0)
cli_arr = clients_df[['XX', 'YY']].to_numpy()
war_arr = warehouses_df[['XX', 'YY']].to_numpy()


""" Debug
print("warehouses")
print(warehouses_df)
print("clients")
print(clients_df)
print(cli_arr)
print(war_arr)
"""

####################

####initial state


 

### Array
### 1st position is the store / warehouse
### 2nd position is the dimension ---- 0 - XX and 1 - YY
print(war_arr[1, ])
print(cli_arr[0,] )
#distance = np.linalg.norm(cli_arr[1,] - cli_arr[0,])
#distance = np.linalg.norm(war_arr[1, ] - cli_arr[0, ])

### In routes store 1 is 1 but in cli_arr is in position 0 

###Initialize Routes Array 

routes = np.empty((16,20))  # replace with empty matrix with 16 rows and then append collumns at the end 
                            # advantage we dont have to go through the array add/drop a client from a warehouse
                            #16 vetores de 20 posições
routes[:] = np.NaN



####To-do 
    # Manually add stores to routes vector and test get total cost function




















