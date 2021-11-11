"""
CSC 510 CT 4 - Informed Search Heuristics with Simple AI
Created Friday, October 8, 2021
Due Sunday, October 10th, 2021

Asignment Prompt
----------------
Define a simple real-world search problem requiring a heuristic solution. You can base the problem on the 8-puzzle (or n-puzzle) problem, 
Towers of Hanoi, or even Traveling Salesman. The problem and solution can be utilitarian or entirely inventive.

Write an interactive Python script (using either simpleAI's library or your resources) that utilizes either Best-First search, Greedy Best First search, 
Beam search, or A* search methods to calculate an appropriate output based on the proposed function. The search function does not have to be optimal nor 
efficient but must define an initial state, a goal state, reliably produce results by finding the sequence of actions leading to the goal state. 

Comment Anchors
---------------
I am using the Comment Anchors extension for Visual Studio Code which utilizes specific keywords
to allow for quick navigation around the file by creating sections and anchor points. Any use
of "anchor", "todo", "fixme", "stub", "note", "review", "section", "class", "function", and "link" are used in conjunction with 
this extension. To trigger these keywords, they must be typed in all caps. 
"""

import random
import itertools
import math
from simpleai.search.models import SearchProblem
from simpleai.search.traditional import astar
from os import system, name


# SECTION - Initial State Generation

def gen_pair(n, m):
    """ Generates a pair of random numbers in bounds (0...n-1, 0...m-1)"""
    return random.randrange(0, n-1), random.randrange(0, m-1)

def random_map_gen(n, m, numPiles):
    """
    Generates a random map of n x m size with numPiles of 'dirt' 

    Arguments:
    - n = x-axis length
    - m = y-axis length

             x - a x i s
          0  1  .  .  .  n-1
    y   0
    |   1
    a   .
    x   .
    i   .
    s   m-1

    

    Return:
    - rand_map: Array of characters corresponding to random map
        - '-' indicates empty space
        - 'D' indicates dirt pile
        - 'R' indicates Roomba position
    - start: (x,y) tuple -> starting location of Roomba
    - piles: [1..numPiles] -> list of (x,y) tuples for dirt locations
    """

    # Generate dirt pile locations
    piles = []
    for i in range(numPiles):

        x, y = gen_pair(n,m)

        if (x,y) in piles:
            added = False

            while not added:
                x,y = gen_pair(n,m)

                if not ((x,y) in piles):
                    added = True

        piles.append((x,y))

    # Generate start location
    x,y = gen_pair(n,m)

    if (x,y) in piles:
        added = False

        while not added:
            x,y = gen_pair(n,m)

            if not((x,y) in piles):
                added = True

    start = (x,y)

    # Create Map List
    rand_map = []
    
    for i in range(m):
        row = []

        for j in range(n):
            if (j,i) in piles:
                row.append('D')
            elif (j,i) == start:
                row.append('R')
            else:
                row.append('-')

        rand_map.append(row)

    return rand_map, start, piles 

# !SECTION - Initial State Generation

# SECTION - Utility Functions

def clearTerminal():
    """Clears the terminal of all text on Windows/MacOs/Linux"""
    
    # For windows
    if name == 'nt':
        _ = system('cls')
    # Mac and linux 
    else:
        _ = system('clear')

''' 
list_to_string and string_to_list were copied from the simpleai library eight_puzzle.py example file
URL: https://github.com/simpleai-team/simpleai/blob/master/samples/search/eight_puzzle.py
'''
def list_to_string(list_):
    return '\n'.join([' '.join(row) for row in list_])

def string_to_list(string_):
    return [row.split(' ') for row in string_.split('\n')]

def gen_permutations(list_):
    """Returns a list of all permutations of provided list"""
    return list(itertools.permutations(list_))

def diagonal_distance(coord1, coord2):
    """
    Calculates the diagonal distance between 2 coordinates

    Arguments:
    - coord1 & coord2 should be tuples of length 2 -> (x,y) coordinates

    Return:
    - distance: float of straighline distance value between two coordinates
    """
    return math.sqrt(((coord2[0] - coord1[0])**2) + ((coord2[1] - coord1[1])**2))

def manhattan_distance(coord1, coord2, straight_cost, diag_cost):
    """
    Calculates the manhattan distance between two coordinates.
    
    C * (dx + dy) + (C2-2 * C) * min(dx, dy)

    Algorithm copied from theory.stanford.edu article on heuristics.
    URL: http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html 

    Arguments:
    - coord1 & coord2 should be tuples of length 2 -> (x,y) coordinates
    - straight_cost: cost of moving in a straight line 1 unit
    - diag_cost: cost of moving in a diagonal line 1 unit

    Return:
    - manhattan_distance: float corresponding to the manhattan distance between the two coordinates
    """
    dx = abs(coord1[0] - coord2[0])
    dy = abs(coord1[1] - coord2[1])
    return straight_cost * (dx + dy) + (diag_cost - 2 * straight_cost) * min(dx, dy)

def find_roomba_location(state_list):
    """Takes a state list and returns the (X,Y) coordinates of the Roomba"""

    for y, row in enumerate(state_list):
        for x, element in enumerate(row):
            if element == 'R' or element == 'X' or element == 'O':
                return x,y

def is_map_clean(map_list):
    """Searches the map to determine if it is clean or not"""
    for y, row in enumerate(map_list):
        for x, element in enumerate(row):
            if element == 'D' or element == 'X':
                return False
    
    return True

def check_clean_waypoints(map_list, waypoint_list):
    """Checks all coordinates in list to determine if they are clean. Returns list of coordinates w/ unlcean state"""

    unclean = []
    for i, coord in enumerate(waypoint_list):
        if map_list[coord[1]][coord[0]] == 'D' or map_list[coord[1]][coord[0]] == 'X':
            unclean.append(coord)

    return unclean

def compute_waypoint_distances(waypoint_list):
    """
    Generates all permutations for provided waypoint list and calculates distances between each successive waypoint combination.
    Distance is calculated using the diagonal_distance formula.

    Please note that this function does not consider all permutations of len(waypoint_list) .. len(1). Permutations and distances
    are calculated assuming length of waypoint list does not change.

    Argument:
    - waypoint_list: List containing coordinate pairs (x,y) of waypoints 

    Return:
    - waypoint_distances: A dictionary containing all waypoint permutations and the corresponding total distance values
    """
    perms = gen_permutations(waypoint_list)

    waypoint_distance = {}

    for path in perms:
        distance = 0

        for i in range(len(path) - 1):
            distance += diagonal_distance(path[i], path[i+1])

        waypoint_distance[path] = distance

    return waypoint_distance


# !SECTION - Utility Functions

class VacuumProblem(SearchProblem):

    def __init__(self, map_list, starting_coords, waypoint_list):

        self.COSTS = {
            'U': 1.0,
            'D': 1.0,
            'R': 1.0,
            'L': 1.0,
            'UL': 1.41,
            'UR': 1.41,
            'DR': 1.41,
            'DL':1.41,
            'S': 2.0
        }

        initial_state = list_to_string(map_list)
        self.waypoints = waypoint_list
        self.starting_coords = starting_coords


        super().__init__(initial_state=initial_state)

    def actions(self, state):
        """Returns a list of possible actions to take at a given location"""
        map_list = string_to_list(state)

        rx, ry = find_roomba_location(map_list)
        m = len(map_list)
        n = len(map_list[0])

        actions = []

        if ry > 0:
            actions.append('U')
        if ry < m-1:
            actions.append('D')
        if rx > 0:
            actions.append('L')
        if rx < n-1:
            actions.append('R')
        if ry > 0 and rx > 0:
            actions.append('UL')
        if ry > 0 and rx < n-1:
            actions.append('UR')
        if ry < m-1 and rx > 0:
            actions.append('DL')
        if ry < m-1 and rx < n-1:
            actions.append('DR')
        
        actions.append('S')

        return actions

    def result(self, state, action):
        """Returns the resulting state after performing the passed action"""
        map_list = string_to_list(state)
        rx,ry = find_roomba_location(map_list)

        if action.count('S'):
            if map_list[ry][rx] == 'X':
                map_list[ry][rx] = 'O'
        else:
            new_rx, new_ry = rx, ry

            if action.count('U'):
                new_ry -= 1
            if action.count('L'):
                new_rx -= 1
            if action.count('D'):
                new_ry += 1
            if action.count('R'):
                new_rx += 1

            if map_list[ry][rx] == 'R':
                map_list[ry][rx] = '-'
            elif map_list[ry][rx] == 'X':
                map_list[ry][rx] = 'D'
            elif map_list[ry][rx] == 'O':
                map_list[ry][rx] = 'C'

            if map_list[new_ry][new_rx] == '-':
                map_list[new_ry][new_rx] = 'R'
            elif map_list[new_ry][new_rx] == 'D':
                map_list[new_ry][new_rx] = 'X'
            if map_list[new_ry][new_rx] == 'C':
                map_list[new_ry][new_rx] = 'O'

        new_state = list_to_string(map_list)
        return new_state

    def is_goal(self, state):
        map_list = string_to_list(state)
        
        if is_map_clean(map_list) and (self.starting_coords == find_roomba_location(map_list)):
            return True

        return False

    def cost(self, state1, action, state2):
        """Returns the cost of performing an action"""
        return self.COSTS[action]

    def heuristic(self, state):
        """Returns an estimation of the distance from a state to the goal"""

        map_list = string_to_list(state)
        r_coords = find_roomba_location(map_list)
        h_vals = []
        #corresponding_distances = {}

        # What are current waypoints?
        dirt_remaining = check_clean_waypoints(map_list, self.waypoints)

        if len(dirt_remaining) > 0:
            waypoint_distances = compute_waypoint_distances(dirt_remaining)

            for i, waypoint in enumerate(dirt_remaining):
            #    for key in waypoint_distances.keys():
            #        if key[0] == waypoint:
            #            corresponding_distances[key] = waypoint_distances[key]

                corresponding_distances = {key:val for key, val in waypoint_distances.items() if key[0] == waypoint}
                min_key = min(corresponding_distances, key=corresponding_distances.get)
                h_vals.append(manhattan_distance(r_coords, waypoint,1,1.41) + corresponding_distances[min_key] + (2 * len(dirt_remaining)) + manhattan_distance(min_key[0], self.starting_coords,1,1.41))
                

            return min(h_vals)

        else:
            return manhattan_distance(r_coords, self.starting_coords, 1, 1.41)

def main():
    
    while True:
        clearTerminal()
        print("* * * * * Vacuum World A* Search * * * * *")
        print("\n")
        print("(1) Info")
        print("(2) Run Program")
        print("(3) Quit")
        print(">> ", end=' ')
        userInput = input()

        if userInput == '1':
            # Print info
            print("\nThis script runs an A* search on a modified Vacuum World problem.")
            print("Norvig and Russell discussed vacuum world in their textbook on Artificial Intelligence. The premise of the initial problem was an agent could exist in one of two locations. ")
            print("Each location could contain the agent, no agent, and/or dirt. The agent was a vacuum that had to move between the two locations and vacuum up the dirt. ")
            print("This problem was the inspiration for the search problem defined in this script.")
            print("\nFor the following script, a grid of n*m locations can be created with a provided number of randomly generated dirt piles. ")
            print("The agent is given a random starting location and has to traverse the optimal path between all piles, vacuum them up, and return to the starting location. ")
            print("The idea was a roomba navigating a room in the most optimal manner and returning to its charging pad.")
            print("\nWhen running the program, the user will be asked to input the size of the 'board' and number of dirt piles to generate. They will then be asked to verify the random board before any search is conducted. ")
            print("The simpleai library was used to implement the VacuumWorld problem and conduct the A* search. ")
            print("IMPORTANT NOTE: Algorithm for waypoint computation is not optimized, so larger board sizes require signifacantly more computaiton time. Recommend grid size no larger than 7*7 w/ 5 dirt piles")
            print("\nUpon completion of the search, the path from start to finish with associated actions will be displayed. The following is a breakdown of the corresponding symbols that will be seen: ")
            print("\tActions: ")
            print("\t\tU = Move Up")
            print("\t\tD = Move Down")
            print("\t\tL = Move Left")
            print("\t\tR = Move Right")
            print("\t\tUL = Move Up and Left (Diagonally)")
            print("\t\tUR = Move Up and Right (Diagonally)")
            print("\t\tDL = Move Down and Left (Diagonally)")
            print("\t\tDR = Move Down and Right (Diagonally)")
            print("\t\tS = Suck")
            print("\n\tStates:")
            print("\t\t - = Empty Tile")
            print("\t\t R = Tile with Roomba")
            print("\t\t D = Tile with Dirt Pile")
            print("\t\t O = Cleaned Tile with Roomba")
            print("\t\t X = Dirty Tile with Roomba")
            print("\t\t C = Cleaned Tile with No Roomba\n\n")
            print("Press any key to continue...",end='')
            input()
        elif userInput == '2':

            clearTerminal()
            print("* * * * * Vacuum World A* Search * * * * *\n")
            print("Please enter a grid height >> ", end='')
            n = int(input())
            print("Please enter a grid width >> ", end='')
            m = int(input())
            print("Please enter number of dirt piles >> ", end='')
            numPiles = int(input())

            confirmed = False
            while not confirmed:
                print(f"Please confirm height={n} width={m}, and number of piles={numPiles} (y|n) >> ", end='')
                userInput = input()
                if userInput.lower() == 'y':
                    confirmed = True
                if userInput.lower() == 'n':
                    clearTerminal()
                    print("* * * * * Vacuum World A* Search * * * * *\n")
                    print("Please enter a grid height >> ", end='')
                    n = int(input())
                    print("Please enter a grid width >> ", end='')
                    m = int(input())
                    print("Please enter number of dirt piles >> ", end='')
                    numPiles = int(input())
                    

            confirmed = False
            while not confirmed:
                clearTerminal()
                print("* * * * * Vacuum World A* Search * * * * *\n")
                rand_map, startCoord, waypoints = random_map_gen(n,m,numPiles)
                print(list_to_string(rand_map))
                print("\nIs this map okay? (y|n) >> ", end='')
                userInput = input()
                if userInput.lower() == 'y':
                    confirmed = True

            print("Beginning Search! Please be patient...")
            result = astar(VacuumProblem(rand_map, startCoord, waypoints))

            for action, state in result.path():
                print('\nAction:', action)
                print(state)

            print("Press any key to continue...",end='')
            input()

        elif userInput == '3':
            quit()
        
if __name__ == "__main__":
    main()


