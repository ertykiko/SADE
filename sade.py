
import pandas as pd
import numpy as np
import random
import math
from timeit import default_timer as timer


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


def order_route(routes,cli_arr,war_arr):
    ##get the smallest distance

    for client in range(len(routes[warehouse])):
        ##Calculate distance to warehouse form all clients
        distance[client] = get_distance(cli_arr[route[warehouse][client]], war_arr[warehouse, ])
        #order by increasing


def get_neighbors(state):
    """Returns neighbors of the argument state for your solution."""
    
        

    raise NotImplementedError


def get_total_cost(routes, cli_arr, war_arr):
      
      cost=0
      for warehouse in range(16):
            for client in range(20):
                if len(routes[warehouse]) == client  : 
                    # Finished the end of the route or no stores in route
                    if client == 0:
                        #route is empty --  no cost
                        break
                    else :
                        #is the last store
                        #calculate cost from last shop to the warehouse
                        print("Client : " + str(client) +
                               " - Last store to warehouse")
                        cost += get_distance( cli_arr[routes[warehouse][client-1],] ,war_arr[warehouse,])
                        print("Reached the end of the warehouse" + str(warehouse))
                        break
                    
                if client == 0:
                    #first store calculate to warehouse
                    #calculate storage to 1st store
                    print("Client : " + str(client) + " - First store to warehouse" )
                    cost += get_distance(war_arr[warehouse, ], cli_arr[routes[warehouse] [client], ])
                else:
                    print("Client : " + str(client))
                    cost += get_distance(cli_arr[routes[warehouse][client], ], cli_arr[routes[warehouse][client-1], ])
                    #cost =+ calculate_cost_between_stores
      
      return cost
    #raise NotImplementedError


def get_distance(location1, location2):
    #Example -- distance = np.linalg.norm(war_arr[1, ] - cli_arr[0, ])
    distance = np.linalg.norm(location1 - location2)
    #print(distance)
    return distance 

def check_routes(routes):
    for warehouse in range(16):
        print(routes[warehouse])

def initialize_routes(routes,war_arr,cli_arr,cli_dem,war_cap):
    vector_stores = list(range(0, 86))
    #random.shuffle(vector_stores)
    #print(vector_stores)
    i = 0
    j = 0
    while i<len(vector_stores):
        if j % 16 == 0 and i != 0:
            j = 0
        print('Checking store ',vector_stores[i], 'with demand ',cli_dem[vector_stores[i]] ,'to append on warehouse ',j, 'with free capacity ', war_cap[j]-sum(cli_dem[routes[j]]))
        if war_cap[j]>=sum(cli_dem[routes[j]])+cli_dem[vector_stores[i]]:
            routes[j].append(vector_stores[i])
            print('Store', vector_stores[i],'with index',vector_stores[i],'attributed to WH',j,'\n')
            i=i+1
        j=j+1 
    war_distance[16]={0}
    for stores in vector_stores:
        for warehouse in range(16):
            if(warehouse == 0 ):
                distance =  get_distance(war_arr[warehouse, ], cli_arr[stores,])
                choice = warehouse
            else :
                new_distance =  get_distance(war_arr[warehouse, ], cli_arr[stores, ])
                if(new_distance < distance):
                    distance = new_distance
                    choice = warehouse
        if(distance < war_distance[choice] or war_distance[choice] == 0) :
            #Do nothing    
            war_distance[choice]=distance

              



#################


##----Import data from CSV----##
start = timer()
clients_df = pd.read_csv("clients.txt",sep=';',header=0)
warehouses_df = pd.read_csv("warehouses.txt",sep=';',header=0)
cli_arr = clients_df[['XX', 'YY']].to_numpy()
cli_dem = clients_df[['DEMAND']].to_numpy()
war_arr = warehouses_df[['XX', 'YY']].to_numpy()
war_cap = warehouses_df[['CAPACITY']].to_numpy()

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
# print(war_arr[1, ])
# print(cli_arr[0,] )
#distance = np.linalg.norm(cli_arr[1,] - cli_arr[0,])
#distance = np.linalg.norm(war_arr[1, ] - cli_arr[0, ])

### In routes store 1 is 1 but in cli_arr is in position 0 

###################################################################################
routes = [[] for i in range(0,16)] 


war_distance = np.zeros(16,int)

print(war_distance)
war_distance[0]=9
print(war_distance)
print(war_distance[0])



#initialize_routes(routes)
#check_routes(routes)


#######################################################################################################

#total_cost = 0

#16 vetores de 20 posições
#total_cost = get_total_cost(routes,cli_arr,war_arr)


#print("Total cost is " + str(total_cost) )

#end = timer()
#print("Time elapsed: " + str(end - start))
####To-do 
    # Manually add stores to routes vector and test get total cost function




















