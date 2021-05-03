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
    for warehouse in range(len(routes)):
        for client in range(len(routes[warehouse])):
            if len(routes[warehouse]) == client+1  : 
                
                #is the last store
                #calculate cost from last shop to the warehouse
                print("CALCULATE DISTANCE FROM CLIENT:",client,'TO CLIENT:',client-1)
                print(get_distance(cli_arr[routes[warehouse][client]], cli_arr[routes[warehouse][client-1]]))
                cost = cost + get_distance(cli_arr[routes[warehouse][client]], cli_arr[routes[warehouse][client-1]])
                print("CALCULATE DISTANCE FROM CLIENT:",client,'WAREHOUSE:',warehouse)
                print(get_distance( cli_arr[routes[warehouse][client]] ,war_arr[warehouse]))
                cost = cost + get_distance( cli_arr[routes[warehouse][client]] ,war_arr[warehouse])
                # print("Reached the end of the warehouse" + str(warehouse))
                break
                    
            if client == 0:
                #first store calculate to warehouse
                #calculate storage to 1st store
                print("CALCULATE DISTANCE FROM WAREHOUSE:",warehouse,'TO CLIENT:',client)
                print(get_distance(war_arr[warehouse], cli_arr[routes[warehouse][client]]))
                cost = cost + get_distance(war_arr[warehouse], cli_arr[routes[warehouse][client]])
            else:
                print("CALCULATE DISTANCE FROM CLIENT:",client,'TO CLIENT:',client-1)
                print(get_distance(cli_arr[routes[warehouse][client]], cli_arr[routes[warehouse][client-1]]))
                cost = cost + get_distance(cli_arr[routes[warehouse][client]], cli_arr[routes[warehouse][client-1]])
                #cost =+ calculate_cost_between_stores
    end = timer()
    # print("Time elapsed: " + str(end - start))
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

    vector_stores = list(range(0, 86)) ##Create list with indexes of stores same during function
    print(vector_stores)
    free_stores=86 ##Number of stores not assigned to a warehouse
    cli_array = np.copy(cli_arr) ##Coordinates of stores
    current_coord =np.copy(war_arr) ##Initial coordinates (warehouses)
    copy_routes= [[] for i in range(0,16)] ##Make a copy of routes to update after an iteration
    
    while free_stores>0 :
        vector_wh= list(range(0, 16)) ##Create list with indexes of warehouses (is created again after an iteration)
        war_cap_array = war_cap ##Warehouse Capacities
        dist_w_s = np.asarray([np.linalg.norm(current_coord[i]-cli_array,axis=1) for i in range(16)]) ##Calculate a 16 by 86 array of the distance of each shop to the warehouses (init) and then between stores available and stores already assigned
        print(np.array(dist_w_s).shape) ##Check dimensions of distance w_s array
        count=np.array(dist_w_s).shape[0] ##Guarantees that in each iteration one store is assigned to every warehouse
        
        while count>0:
=======
    #     if war_cap[j]>=sum(cli_dem[routes[j]])+cli_dem[vector_stores[i]]:
    #         routes[j].append(vector_stores[i])
    #         print('Store', i,'with index',vector_stores[i],'attributed to WH',j,'\n')
    #         i=i+1
    #     j=j+1 

    free_stores=86
    cli_array = cli_arr

    current_coord = war_arr
    while free_stores>0 : 
        dist_w_s = np.asarray([np.linalg.norm(current_coord[i]-cli_array,axis=1) for i in range(16)]) ##Calculate a 16 by 86 array of the distance of each shop to the warehouses
        #print("Dist Warehouse to Store ")
        print(np.array(dist_w_s).shape) ##Check dimensions of distance w_s array
        i=0
        while i<16:
            closest_store = np.argmin(dist_w_s[i,:]) #Check the closest store for the give warehouse -- i
            print("Closest store : " + str(closest_store))
            if war_cap[i]>=sum(cli_dem[routes[i]])+cli_dem[vector_stores[closest_store]]: #Check if capacity allows for store to be assigned to warehouse
                store_index = vector_stores.pop(closest_store)  ## Remove 
                routes[i].append(store_index)  ##append store to warehouse route
                free_stores=free_stores-1 ##One less free store
                dist_w_s = np.delete(dist_w_s,closest_store,1)  ##Delete distance from w_s array
                current_coord[i]=cli_array[closest_store] 
                cli_array= np.delete(cli_array,closest_store,0) ##Delete form cli_array
            else: print('Not enough capacity')
             
            if free_stores==0:break
            
            i=i+1
            
      



##----Import data from CSV----##
start = timer()
clients_df = pd.read_csv("clients.txt",sep=';',header=0)
warehouses_df = pd.read_csv("warehouses.txt",sep=';',header=0)
cli_arr = clients_df[['XX', 'YY']].to_numpy()
cli_dem = clients_df[['DEMAND']].to_numpy()
war_arr = warehouses_df[['XX', 'YY']].to_numpy()
war_id = warehouses_df[['ID']].to_numpy()
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

initialize_routes(routes,war_arr,cli_arr,cli_dem,war_cap)
#check_routes(routes)

#war_arr = warehouses_df[['XX', 'YY']].to_numpy() ##Get the values again of the coordinates of the warehouses
#######################################################################################################
#total_cost = 0

#16 vetores de 20 posições
total_cost = get_total_cost(routes,cli_arr,war_arr)


print("Total cost is " + str(total_cost) )

end = timer()

for x in range(16):
     print(war_cap[x]-sum(cli_dem[routes[x]]))
print("Time elapsed: " + str(end - start))


print("Routes :")
check_routes(routes)
####To-do 
    # Manually add stores to routes vector and test get total cost function

















