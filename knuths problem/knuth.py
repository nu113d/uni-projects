import math
import time
#from factorial_calc import factorial

INITIAL = 4
FACTORIAL_MAX = 500000 #Do not compute factorial if it's larger than this. If None the value is by default 2147483647

def expand(item):
    integer_item = int(item)
    root = factorial = 0

    try:
        if item < 1:  #there is no point searching below 1
            raise OverflowError
        root = math.sqrt(integer_item)            
    except OverflowError: #too many digits
        root = None  

    if item - abs(integer_item) < 0.00000001: #check if it is an integer
        #catch exceptions because numbers can get reaaaaally big
        try:
            if FACTORIAL_MAX and integer_item > FACTORIAL_MAX:
                raise OverflowError
            factorial = math.factorial(integer_item)
        except OverflowError: #factorial too big
            factorial = None      

        return [{ 'action': 'root', "value":root, 'prev': item},
                {"action": 'factorial', 'value': factorial, 'prev': item }]  #2 new branches with root and factorial left and right respectively, factorial and sqrt are None if they are too big or there is no point searching any more
       
                      
    else:
        return [{ 'action': 'root', "value": root, 'prev': item},
                {"action": 'floor', 'value': math.floor(item), 'prev': item }]
 

def breadth(INITIAL, GOAL):
    frontier = []
    frontier.append({'action':None, 'value': INITIAL, 'prev': None}) #The root of the tree

    i = 0
    while i < len(frontier):
        if frontier[i]['value']==GOAL:
            print("PROBLEM SOLVED!!")
            print("Initial value (root): ", INITIAL)
            find_actions(frontier)
            return
        if frontier[i]['value'] == None:
            i+=1
            continue  #do not expand the item because it's not computed
        nodes_to_add = expand(frontier[i]['value'])

        for j in nodes_to_add:
            if j not in frontier:
                frontier.append(j)
            if j['value']==GOAL: #check solution also when adding a new node
                print("PROBLEM SOLVED!!")
                print("Initial value (root): ", INITIAL)
                find_actions(frontier)
                return
        i+=1
    if i == len(frontier): #frontier is empty, no solution
        print("No solution found. Try increasing FACTORIAL_MAX")

def iddfs(INITIAL , GOAL):
    max_depth = 0
    while True:
        frontier = []
        frontier.append({'action':None, 'value': INITIAL, 'prev': None, 'depth':0}) #The root of the tree
        current_depth = 0
        while frontier: 
            item = frontier.pop(0)         
            if item['value']==GOAL:
                print("PROBLEM SOLVED!!")
                print("Initial value (root): ", INITIAL)
                print("Depth: ", item['depth'])
                return
            if item['value'] == None or item['depth']==max_depth: 
                continue #do not expand the item because it's not computed or reached the max_depth
            nodes_to_add = expand(item['value'])
            current_depth = item['depth'] + 1

            for j in nodes_to_add:
                if j not in frontier:
                    j['depth'] = current_depth  #Add a new key to keep the depth of the node
                    frontier.insert(0, j)
        max_depth+=1

def find_actions(frontier):
    actions_list = []
    actions_list.append(frontier[-1]['action']) #begin from the last node and search backwards
    prev_node = frontier[-1]['prev']

    while prev_node:
        for current_node in frontier:
            if current_node['value']==prev_node:
               prev_node=current_node['prev']
               actions_list.append(current_node['action'])
               break
    
    actions_list.pop()  # remove the last item because it's always None
    print("Depth: ", len(actions_list))
    print("Actions:")
    for i in reversed(actions_list):  #print actions_list in reverse order
        print("\t", i)
    
    


def main():
    GOAL = int(input("Enter the integer you want to search: "))
    print("Choose the search algorithm \n\t1.breadth-first\n\t2.iterative deepening")
    ALG = int(input(">"))
    start_time = time.time()
    if ALG == 1:
        breadth(INITIAL, GOAL)
    elif ALG==2:
        iddfs(INITIAL, GOAL)
    else:
        print("invalid choice")
        exit(1)    
    print("Goal: ", GOAL)
    print(time.time() - start_time, "s")
    


if __name__ == "__main__":
    main()