import pandas as pd
import numpy as np
import random
import math
import copy
from timeit import default_timer as timer


###-----Simulated Annealing

def simulated_annealing(initial_route,cli_arr,cli_dem,war_cap,war_arr):
    """Peforms simulated annealing to find a solution"""
    initial_temp = 100
    final_temp = .1
    alpha = 0.001

    current_temp = initial_temp

    # Start by initializing the current state with the initial state
    current_route = initial_route
    solution = current_route
    
    iteration=1

    while current_temp > final_temp:
        #neighbor = random.choice(get_neighbors())
        rand_func = np.random.randint(0,4) #Chose randomly between swap and append
        
        if rand_func==0 :
            print("Append")
            neighbor = get_neighbors_random_append(solution,cli_dem,war_cap,war_arr,cli_arr)
        else:
            print("Swap")
            neighbor = random.choice(get_neighbors_random_swap(solution,cli_dem,war_cap,war_arr,cli_arr))
        
        # Check if neighbor is best so far
        cost_diff = get_total_cost(neighbor,cli_arr,war_arr) - get_total_cost(solution,cli_arr,war_arr)
        print("----------------------------------------------------COST DIFF: ",cost_diff)  
        print("Neighbor cost is : " + str(get_total_cost(neighbor, cli_arr, war_arr)))

        # if the new solution is better, accept it
        if cost_diff < 0:
            print('Improved')
            #print("sol cost is : " + str(get_total_cost(solution, cli_arr, war_arr)))
            solution = neighbor
            #print("Neighbor cost is : " + str(get_total_cost(neighbor, cli_arr, war_arr)))
            #break
        # if the new solution is not better, accept it with a probability of e^(-cost/temp)
        else:
            rand = random.uniform(0,1)
            mathio = math.exp(-cost_diff / current_temp)

            #print("cost_diff", cost_diff)
            #print("CT: ",current_temp)
            print("RAND:",rand," >  MATH:",mathio)
		    
		    #t = temp / float(i + 1)
        
            
            if rand > mathio:
                
                print("Accepting Worse Solution")
                solution = neighbor
        
        # decrement the temperature
        current_temp -= alpha
        iteration += 1

    print("Finished Simulated Annealing")
    return solution

def get_neighbors_random_swap(route,cli_dem,war_cap,war_arr,cli_arr):
    """Returns neighbors of the argument state for your solution."""
    new_route=copy.deepcopy(route) #Is this copy necessary ?
    ##Find target warehouse
    target_warehouse = np.random.randint(0,16,2)
    print(target_warehouse)
    print(route[target_warehouse[0]])
    index_s0 = np.random.randint(0,len(route[target_warehouse[0]]))  ##Bug here -- Need to check if the routes is not empty
    index_s1 = np.random.randint(0,len(route[target_warehouse[1]]))
    target_store0 = route[target_warehouse[0]][index_s0]
    target_store1 = route[target_warehouse[1]][index_s1]
    
    print("index store 0",index_s0,"  ",target_store0)
    print("index store 1",index_s1,"  ",target_store1)
    
    if swap_stores(target_warehouse[0],target_warehouse[1],index_s0,index_s1,cli_dem,war_cap,new_route):
        check_order(new_route[target_warehouse[0]],war_arr[target_warehouse[0]],cli_arr)
        check_order(new_route[target_warehouse[1]],war_arr[target_warehouse[1]],cli_arr)
              
    
    return new_route

def get_neighbors_random_append(route,cli_dem,war_cap,war_arr,cli_arr):
    new_route=copy.deepcopy(route)
    ##Find target warehouse
    target_warehouse = np.random.randint(0,16,2)
    print(target_warehouse)
    print(route[target_warehouse[0]])
    index_s0 = np.random.randint(0,len(route[target_warehouse[0]])) ##Shotty as fack
    target_store0 = route[target_warehouse[0]][index_s0]

    if war_cap[target_warehouse[1]] >= sum(cli_dem[routes[target_warehouse[1]]])+cli_dem[target_store0]:
        store = new_route[target_warehouse[0]].pop(index_s0)
        new_route[target_warehouse[1]].append(store)
        check_order(new_route[target_warehouse[0]],war_arr[target_warehouse[0]],cli_arr)
        check_order(new_route[target_warehouse[1]],war_arr[target_warehouse[1]],cli_arr)
    return new_route


def swap_stores(warehouse1,warehouse2,store_pos1,store_pos2,cli_dem,war_cap,route):
    swapper = 0
    print('WH1', warehouse1, 'WH2', warehouse2, 'ST1', store_pos1, 'ST2', store_pos2)

    if ( (war_cap[warehouse1] - sum(cli_dem[route[warehouse1]]) - cli_dem[route[warehouse2][store_pos2]] + cli_dem[route[warehouse1][store_pos1]]) >= 0 and  
    ( war_cap[warehouse2] - sum(cli_dem[route[warehouse2]]) - cli_dem[route[warehouse1][store_pos1]] + cli_dem[route[warehouse2][store_pos2]] ) >= 0 ):
        #Swap is good 
        swapper = route[warehouse1][store_pos1]
        route[warehouse1][store_pos1] = route[warehouse2][store_pos2]
        route[warehouse2][store_pos2] = swapper
        return 1
    else :
        return 0

def get_total_cost(routes, cli_arr, war_arr):
    
    cost = 0
    for warehouse in range(len(routes)):
        print(len(routes))
        if( routes[warehouse] is not None):
            for client in range(len(routes[warehouse])):
                if len(routes[warehouse]) == client+1:

                    #is the last store
                    #calculate cost from last shop to the warehouse
                    #print("CALCULATE DISTANCE FROM CLIENT:",
                    #      client, 'TO CLIENT:', client-1)
                    #print(get_distance(
                    #    cli_arr[routes[warehouse][client]], cli_arr[routes[warehouse][client-1]]))
                    cost = cost + \
                        get_distance(cli_arr[routes[warehouse][client]],
                                    cli_arr[routes[warehouse][client-1]])
                    #print("CALCULATE DISTANCE FROM CLIENT:",
                    #      client, 'WAREHOUSE:', warehouse)
                    #print(get_distance(
                    #    cli_arr[routes[warehouse][client]], war_arr[warehouse]))
                    cost = cost + \
                        get_distance(cli_arr[routes[warehouse]
                                    [client]], war_arr[warehouse])
                    # print("Reached the end of the warehouse" + str(warehouse))
                    break

                if client == 0:
                    #first store calculate to warehouse
                    #calculate storage to 1st store
                    #print("CALCULATE DISTANCE FROM WAREHOUSE:",
                    #      warehouse, 'TO CLIENT:', client)
                    #print(get_distance(war_arr[warehouse],
                    #      cli_arr[routes[warehouse][client]]))
                    cost = cost + \
                        get_distance(war_arr[warehouse],
                                    cli_arr[routes[warehouse][client]])
                else:
                    #print("CALCULATE DISTANCE FROM CLIENT:",
                    #      client, 'TO CLIENT:', client-1)
                    #print(get_distance(
                    #    cli_arr[routes[warehouse][client]], cli_arr[routes[warehouse][client-1]]))
                    cost = cost + \
                        get_distance(cli_arr[routes[warehouse][client]],
                                    cli_arr[routes[warehouse][client-1]])
                    #cost =+ calculate_cost_between_stores
        else :
            print("warehouse is empty skip it ")
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


def get_free_space(routes,war_cap,cli_dem):
    free_space = np.array([])
    for x in range(16):
        #print(war_cap[x]-sum(cli_dem[routes[x]]))
        free_space = np.append(free_space, (war_cap[x]-sum(cli_dem[routes[x]])))
    return free_space

def check_order(warehouse,war_arr,cli_arr):
    vector_stores = list(warehouse)
    print("route",vector_stores)
    cli_array = np.copy(cli_arr) ##Coordinates of stores
    current_coord =np.copy(war_arr)
    order_route = [[] for i in range(0)]
    dist_w_s = np.asarray([np.linalg.norm(current_coord-cli_array[vector_stores[i]],axis=0) for i in range(len(vector_stores))])
   # print("distws", dist_w_s)

    count = len(dist_w_s)
    #print(count)
    while count>0:
        dist_w_s = np.asarray([np.linalg.norm(current_coord-cli_array[vector_stores[i]],axis=0) for i in range(len(vector_stores))])
        #print("distws", dist_w_s)
        y =int(np.where(dist_w_s == np.amin(dist_w_s))[0])
        store_index = vector_stores.pop(y) ##Remove index of Store to assign
        order_route.append(store_index)  ##Append store to warehouse route 
        dist_w_s = np.delete(dist_w_s,y) ##Delete column Store from dist_w_s
        current_coord=cli_array[y] ##New coordinates from Store assigned to calculate distances with Stores available in the next iteration 
        count=count-1
    print("new_route:",order_route)
    return order_route

def initialize_routes(routes, war_arr, cli_arr, cli_dem, war_cap):

    # Create list with indexes of stores same during function
    vector_stores = list(range(0, 86))
    #print(vector_stores)
    free_stores = 86  # Number of stores not assigned to a warehouse
    cli_array = np.copy(cli_arr)  # Coordinates of stores
    current_coord = np.copy(war_arr)  # Initial coordinates (warehouses)
    # Make a copy of routes to update after an iteration
    war_cap_array = war_cap
    recalculate_distances = True
    while free_stores > 0:
          # Warehouse Capacities
        # Calculate a 16 by 86 array of the distance of each shop to the warehouses (init) and then between stores available and stores already assigned
        if recalculate_distances:
            dist_w_s = np.asarray([np.linalg.norm(current_coord[i]-cli_array, axis=1) for i in range(16)])
        # Check dimensions of distance w_s array
        #print(np.array(dist_w_s).shape)

        # X (Warehouse) index of the best solution of all possibilities
        x = np.unravel_index(dist_w_s.argmin(), dist_w_s.shape)[0]
        # Y (Store) index of the best solution of all possibilities
        y = np.unravel_index(dist_w_s.argmin(), dist_w_s.shape)[1]
        print("Best possibility is store: ",
            vector_stores[y], "   to warehouse:", x+1, '-----> Dist:', np.min(dist_w_s))
        print('Checking store ', y, 'with demand ', cli_dem[vector_stores[y]], 'to append on warehouse ', x, 'with capacity ', war_cap_array[x], 'and the sum of stores: ', sum(
            cli_dem[routes[x]]), 'therefore free:', war_cap_array[x]-sum(cli_dem[routes[x]]))

        # If there is not a best solution that fits the Warehouses exits loop
        if np.all(dist_w_s == dist_w_s[0, 0]) == 1:
                break

            # Check if capacity allows for store to be assigned to warehouse
        if war_cap_array[x] >= sum(cli_dem[routes[x]])+cli_dem[vector_stores[y]]:
                # Remove index of Store to assign
            store_index = vector_stores.pop(y)
                # Remove index of Warehouse to assign
                #wh_index = vector_wh.pop(x)
                # Append store to warehouse route
            routes[x].append(store_index)
            free_stores = free_stores-1  # One less free store
                # Delete column Store from dist_w_s
            dist_w_s = np.delete(dist_w_s, y, 1)
                # Delete line Warehouse from dist_w_s
                #dist_w_s = np.delete(dist_w_s, x, 0)
                # New coordinates from Store assigned to calculate distances with Stores available in the next iteration
            current_coord[x] = cli_array[y]
                # Delete index form cli_array (Store)
            cli_array = np.delete(cli_array, y, 0)
                # Delete index form war_cap_array (Warehouse)
                #war_cap_array = np.delete(war_cap_array, x, 0)
                #count = count-1  # One Warehouse with a assigned store, go to next iteration until every 16 Warehouses have a store assigned
            #del copy_routes[x]  # Delete element from copy_routes
            recalculate_distances = True
        else:
            # Capacity doesn't allow to assign Store
            print('Not enough capacity')
            recalculate_distances = False
            dist_w_s[x][y] = 10000  # Go find next best solution

        if free_stores == 0:
            break  # No more stores to assign --> end of iterations
        print(routes)





##----Import data from CSV----##
start = timer()
clients_df = pd.read_csv("clients.txt", sep=';', header=0)
warehouses_df = pd.read_csv("warehouses.txt", sep=';', header=0)


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
routes = [[] for i in range(0, 16)]

initialize_routes(routes, war_arr, cli_arr, cli_dem, war_cap)


#check_routes(routes)

#war_arr = warehouses_df[['XX', 'YY']].to_numpy() ##Get the values again of the coordinates of the warehouses
#######################################################################################################
check_routes(routes)
#print("Lenght of warehouse route 1 is " + str(len(routes[0])))

initial_cost = get_total_cost(routes, cli_arr, war_arr)


print("Initial cost is " + str(initial_cost))

final_routes = simulated_annealing(routes,cli_arr,cli_dem,war_cap,war_arr)

final_cost = get_total_cost(final_routes, cli_arr, war_arr)

print("Final cost is " + str(final_cost))

end = timer()
print("Time elapsed: " + str(end - start))

print(get_free_space(routes,war_cap,cli_dem))

####To-do
# Manually add stores to routes vector and test get total cost function
