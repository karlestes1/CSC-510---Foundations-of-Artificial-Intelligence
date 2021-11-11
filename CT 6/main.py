"""
Karl Estes
CSC classNumber Assignment
Created: today
Due: dueDate
"""
import pandas as pd
import numpy as np
import math
from os import system, name
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB


def round_dec(num, places = 2):
    """Rounds the provided number to a specific number of decimal places (defaulted to 2)"""

    multiplier = 10 ** places

    return math.trunc(num * multiplier) / multiplier


def clear_terminal():
    """Clears the terminal of all text on Windows/MacOs/Linux"""
    
    # For windows
    if name == 'nt':
        _ = system('cls')
    # Mac and linux 
    else:
        _ = system('clear')

def convert_to_categorical(data):
    '''
    Converts the iris data to categorical based on the following scheme:
    
    - Sepal Length:
        - Short = 4.3-5.5
        - Medium = 5.51-6.7
        - Long = 6.71-7.9
    - Sepal Width:
        - Thin = 2.0-2.8
        - Medium = 2.81-3.6
        - Wide = 3.61-4.4
    - Petal Length:
        - Short = 1.0-2.97
        - Medium = 2.98-4.94
        - Long = 4.95-6.91
    - Petal Width:
        - Thin = 0.1-0.9
        - Medium = 0.91-1.7
        Wide = 1.71-2.5

    CONVERSION IS DONE INPLACE
    '''

    # Rename Columns
    data.rename(columns={"sepal length (cm)": "SL", "sepal width (cm)": "SW",
                       "petal length (cm)": "PL", "petal width (cm)": "PW",
                       "target": "Class"},inplace=True)
    
    # Replace Class Names
    data['Class'].replace({0: 'Iris Setosa', 1: 'Iris Vericolour', 2: 'Iris Virginica'}, inplace=True)

    # Replace numerical data with categorical
    replacements = [[('Short',4.3,5.5),('Medium',5.51,6.7),('Long',6.71,7.9)],
                    [('Thin',2.0,2.8),('Medium',2.81,3.6),('Wide',3.61,4.4)],
                    [('Short',1.0,2.97),('Medium',2.98,4.94),('Long',4.95,6.91)],
                    [('Thin',0.1,0.9),('Medium',0.91,1.7),('Wide',1.71,2.5)]]

    for column, i in zip(['SL', 'SW', 'PL', 'PW'], range(4)):
        masks = []
        for j in range(3):
            masks.append((data[column] >= replacements[i][j][1]) & (data[column] <= replacements[i][j][2]))

        for mask, j in zip(masks, range(3)):    
            data.loc[mask, column] = replacements[i][j][0]

def create_likelihood_tables(data):
    '''Generates the likelihood tables for the data'''
    tables = []
    
    for column in ['SL', 'SW', 'PL', 'PW']:

        table = pd.crosstab(data[column], data['Class'], margins=True, margins_name='Total')
        # Search for zero value and apply laplace accordingly
        if (0 in table.values):
            table.loc[:table.index[2],:table.columns[2]] += 1
            table.loc['Total', :table.columns[2]] += 3
            table.loc[:table.index[2],'Total'] += 3
            table['Total']['Total'] += 9
        # Add row for P(hypothesis)
        table.loc['P(h)'] = [round_dec((table.loc['Total'][0])/(table.loc['Total'][3]),2),
                             round_dec((table.loc['Total'][1])/(table.loc['Total'][3]),2),
                             round_dec((table.loc['Total'][2])/(table.loc['Total'][3]),2), '']

        # Add Column for P(Data)
        table['P(D)'] = [round_dec((table['Total'][0]/table['Total'][3]),2),
                         round_dec((table['Total'][1]/table['Total'][3]),2),
                         round_dec((table['Total'][2]/table['Total'][3]),2),'','']

        # Add columns for posterior probability
        table['P(D|Iris Setosa)'] = [round_dec(((table['Iris Setosa'][0])/table['Iris Setosa']['Total']),2),
                                                     round_dec(((table['Iris Setosa'][1])/table['Iris Setosa']['Total']),2),
                                                     round_dec(((table['Iris Setosa'][2])/table['Iris Setosa']['Total']),2),'','']
                                                    
        table['P(D|Iris Vericolour)'] = [round_dec(((table['Iris Vericolour'][0])/table['Iris Vericolour']['Total']),2),
                                                     round_dec(((table['Iris Vericolour'][1])/table['Iris Vericolour']['Total']),2),
                                                     round_dec(((table['Iris Vericolour'][2])/table['Iris Vericolour']['Total']),2),'','']

        table['P(D|Iris Virginica)'] = [round_dec(((table['Iris Virginica'][0])/table['Iris Virginica']['Total']),2),
                                                     round_dec(((table['Iris Virginica'][1])/table['Iris Virginica']['Total']),2),
                                                     round_dec(((table['Iris Virginica'][2])/table['Iris Virginica']['Total']),2),'','']
    

        tables.append(table)



    return tables

def run_nb_predictions(tables, features, classes):
    preds = []

    # Calculate Probabilities
    for target in classes:
        probs = []
        for i, feature in enumerate(features):
            probs.append(tables[i][f'P(D|{target})'][feature])

        product = tables[0][target]['P(h)']

        for item in probs:
            product *= item

        preds.append(product)
    
    # Normalize
    s = sum(preds)
    preds = [x / s for x in preds]

    return preds


def nb_interactive(tables):

    # Init w/ min values from dataset
    features = ['Short','Thin','Short','Thin']
    classes = ['Iris Setosa', 'Iris Vericolour', 'Iris Virginica']
    attributes = [('Thin','Medium','Wide'),('Short','Medium','Long')]


    while True:
        clear_terminal()
        print("* * * * * Critical Thinking 6 * * * * *")
        print("     Simple Naïve Bayes Classifier\n")
        print("(v): View Probability Tables")
        print("(q): Quit to Main Menu\n")

        # Menu and current values
        print(f"(1) Sepal Length: {features[0]}")
        print(f"(2) Sepal Width: {features[1]}")
        print(f"(3) Petal Length: {features[2]}")
        print(f"(4) Petal Width: {features[3]}\n")

        # Run prediction
        preds = run_nb_predictions(tables, features, classes)

        print(f"Predicted Class Probabilities: {preds}")
        print(f"Predicted Class: {classes[preds.index(max(preds))]}")

        user_input = input("\n>> ")

        if user_input == '1' or user_input == '2' or user_input == '3' or user_input == '4':
            if (int(user_input) % 2 == 1):
                print("\t(1) Short\n\t(2) Medium\n\t(3) Long")
            else:
                print("\t(1) Thin\n\t(2) Medium\n\t(3) Wide")
            repeat = True
            while repeat:
                try:
                    choice = int(input(f"Please choose new value for feature ({user_input}) >> "))
                    
                    if choice > 0 and choice < 4:
                        features[int(user_input)-1] = attributes[(int(user_input) % 2)][choice-1]
                        repeat = False
                    else:
                        print("Not a volid choice. Try again! . . .")
                except ValueError:
                    print("Not a valid input. Try again! . . .")
        elif user_input.lower() == 'v':
            print("* * Likelihood Tables * *\n")
            print(f"* SEPAL LENGTH *\n{tables[0]}\n")
            print(f"* SEPAL WIDTH *\n{tables[1]}\n")
            print(f"* PETAL LENGTH *\n{tables[2]}\n")
            print(f"* PETAL WIDTH *\n{tables[3]}\n")
            input("Press any key to continue . . .")
        elif user_input.lower() == 'q':
            return

def naive_bayes_simple():

    # Load Data
    df = pd.DataFrame(load_iris(as_frame=True).frame)

    # Convert to Categorical
    convert_to_categorical(df)

    # Generate Likelihood Tables
    tables = create_likelihood_tables(df)

    # User-Interactive Prediction w/ Table Display
    nb_interactive(tables)
    

def gnb_interactive(gnb):

    # Init w/ min values from dataset
    features = [4.3,2.0,1.0,0.1]

    classes = ['Iris Setosa', 'Iris Vericolour', 'Iris-Virginica']

    while True:
        clear_terminal()
        print("* * * * * Critical Thinking 6 * * * * *")
        print("    Gaussian Naïve Bayes Classifier\n")
        print("(q): Quit to Main Menu\n")

        # Menu and current values
        print(f"(1) Sepal Length: {features[0]}")
        print(f"(2) Sepal Width: {features[1]}")
        print(f"(3) Petal Length: {features[2]}")
        print(f"(4) Petal Width: {features[3]}\n")

        # Run prediction
        prob = gnb.predict_proba([[features[0], features[1], features[2], features[3]]])
        class_index = np.argmax(prob[0])

        print(f"Predicted Class Probabilities: {prob[0]}")
        print(f"Predicted Class: {classes[class_index]}")

        user_input = input("\n>> ")

        if user_input == '1' or user_input == '2' or user_input == '3' or user_input == '4':
            repeat = True
            while repeat:
                try:
                    new_val = float(input(f"Please enter a new value for feature ({user_input}) >> "))
                    new_val = round_dec(new_val, 1)
                    features[int(user_input) - 1] = new_val
                    repeat = False
                except ValueError:
                    print("Not a valid input. Try again! . . .")
        elif user_input == 'q':
            return

def naive_bayes_gaussian():

    # Load Data
    X,y = load_iris(return_X_y=True)

    # Train Predictor
    gnb = GaussianNB()
    gnb.fit(X,y)

    # User-Interactive Prediction
    gnb_interactive(gnb)



if __name__ == "__main__":

    while(True):
        clear_terminal()
        print("* * * * * Critical Thinking 6 * * * * *\n")
        print("(1) Naïve Bayes with Categorical Data")
        print("(2) Gaussian Naïve Bayes with Numerical Data")
        print("(3) Exit Program")
        print(">> ", end="")

        user_input = input()

        if user_input == '1':
            naive_bayes_simple()
        elif user_input == '2':
            naive_bayes_gaussian()
        elif user_input == '3':
            quit()
