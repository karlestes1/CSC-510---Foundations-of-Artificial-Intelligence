"""
Karl Estes
CSC 510 Portfolio Project
Created: today
Due: November 7th, 2021

Comment Anchors
---------------
I am using the Comment Anchors extension for Visual Studio Code which utilizes specific keywords
to allow for quick navigation around the file by creating sections and anchor points. Any use
of "anchor", "todo", "fixme", "stub", "note", "review", "section", "class", "function", and "link" are used in conjunction with 
this extension. To trigger these keywords, they must be typed in all caps. 
"""

import tensorflow as tf 
import numpy as np
import pandas as pd
import os
import math
import json
import cv2 as cv
import matplotlib.pyplot as plt
from os import system, name


# Changes the working directory to whatever the parent directory of the script executing the code is
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def clear_terminal():
    """Clears the terminal of all text on Windows/MacOs/Linux"""
    
    # For windows
    if name == 'nt':
        _ = system('cls')
    # Mac and linux 
    else:
        _ = system('clear')

def round_dec(num, places = 2):
    """Rounds the provided number to a specific number of decimal places (defaulted to 2)"""

    multiplier = 10 ** places

    return math.trunc(num * multiplier) / multiplier

class NaiveBayesianClassifier():

    def __init__(self) -> None:
        
        # Load list of likelihood variables
        try:
            print("Loading conversion data for bayesian classification. . .")
            fp = open('Data/conversions.json')
            self.var_dict = json.load(fp)
            self.variables = self.var_dict.keys()
            self.pred_variables = ['_STATE', 'SEXVAR', 'GENHLTH', 'PHYSHLTH', 'MENTHLTH', 'POORHLTH', 'EXERANY2', 'SLEPTIM1', 
                                   'CVDINFR4', 'CVDCRHD4', 'CVDSTRK3', 'ASTHMA3', 'CHCSCNCR', 'CHCOCNCR', 'CHCCOPD2', 'HAVARTH4', 'ADDEPEV3', 
                                   'CHCKDNY2', 'DIABETE4', 'PREGNANT', 'DEAF', 'BLIND', 'DIFFWALK', 'DIFFDRES', 'DIFFALON', 'SMOKE100', 
                                   'USENOW3', 'FALL12MN', 'HIVRISK5', 'TOLDHEPC', 'HAVEHEPB', 'ACEDEPRS', 
                                   'ACEHURT1', 'ACETOUCH', 'ACETTHEM', 'ACEHVSEX', '_RACE', '_AGE_G', '_BMI5CAT', '_RFDRHV7']
        except:
            print("ERROR: Unable to load conversion list for NaiveBayesianClassifier()")
            exit(0)

        # Load all likelihood tables
        print("Loading likelihood tables")
        self.tables = []
        for var in self.pred_variables:
            try:
                self.tables.append(pd.read_csv(os.path.join("Data/tables", f"{var}.csv")))
            except:
                if not (var in ['DECIDE', 'CIMEMLOS', 'CDASSIST', 'CDSOCIAL']):
                    print(f"WARNING: Unable to load likelihood table for {var}")
                    self.tables.append(None)

        self.post_prob_columns = ['P_D_ZERO', 'P_D_ONE', 'P_D_TWO']

    def predict(self, features):
        '''
        Features must appear in the following order:

        '_STATE', 'SEXVAR', 'GENHLTH', 'PHYSHLTH', 'MENTHLTH', 'POORHLTH', 'EXERANY2', 'SLEPTIM1', 
        'CVDINFR4', 'CVDCRHD4', 'CVDSTRK3', 'ASTHMA3', 'CHCSCNCR', 'CHCOCNCR', 'CHCCOPD2', 'HAVARTH4', 'ADDEPEV3', 
        'CHCKDNY2', 'DIABETE4', 'PREGNANT', 'DEAF', 'BLIND', 'DIFFWALK', 'DIFFDRES', 'DIFFALON', 'SMOKE100', 
        'USENOW3', 'FALL12MN', 'HIVRISK5', 'TOLDHEPC', 'HAVEHEPB', 'CIMEMLOS', 'ACEDEPRS', 
        'ACEHURT1', 'ACETOUCH', 'ACETTHEM', 'ACEHVSEX', '_RACE', '_AGE_G', '_BMI5CAT', '_RFDRHV7'

        The features array may have none values, in which case the variable will not be included in the final result (WILL NOT PROVIDE ACCURATE ANSWER)

        A shorter feature array will result in later variable being removed from analysis. 
        '''
        preds = []
        excluded_variables = []

        if len(features) != len(self.pred_variables):
            print(f"WARNING: Mismatch in list length\nProvided list of responses for classification is of length {len(features)} while variables are of length {len(self.variables)}")

        for target in self.post_prob_columns:
            probs = []

            for i,feature in enumerate(features):

                # Check if feature is missing or feature class was present in the dataset
                if (not (feature is None)) or (str(feature) in self.var_dict[self.pred_variables[i]].keys()):
                    
                    # Find row index
                    condition = self.tables[i][self.pred_variables[i]] == (str(feature) + ".0")
                    indices = self.tables[i].index[condition]
                    
                    if indices.shape == (0,):
                        excluded_variables.append(self.pred_variables[i]) # Add excluded variable to exclusion list
                    else:
                        probs.append(list(self.tables[i][target][indices])[0])
                else:
                    excluded_variables.append(self.pred_variables[i]) # Add excluded variable to exclusion list

            if target == 'P_D_ZERO':
                product = self.tables[0]['0'][19]
            elif target == 'P_D_ONE':
                product = self.tables[0]['1'][19]
            elif target == 'P_D_TWO':
                product = self.tables[0]['2'][19]

            for item in probs:
                product *= item

            preds.append(product)

        # Normalize
        s = sum(preds)
        preds = [x / s for x in preds]

        return preds, excluded_variables         

class MRIClassifier():

    def __init__(self) -> None:

        self.cdr = [0,0.5,1,2,3]
        print("Loading saved CNN models for MRI classifications")
        
        # Load each of the classifier models
        try:
            self.cor_model = tf.keras.models.load_model("Data/models/cor_cnn.hdf5")
            print("Loaded coronal classification model")
        except:
            print("ERROR: Issue loading coronal imaging classification model")
        
        try:
            self.tra_model = tf.keras.models.load_model("Data/models/tra_cnn.hdf5")
            print("Loaded transverse classification model")
        except:
            print("ERROR: Issue loading transverse imaging classification model")
        
        try:
            self.sag_model = tf.keras.models.load_model("Data/models/sag_cnn.hdf5")
            print("Loaded sagittal classification model")
        except:
            print("ERROR: Issue loading sagittal imaging classification model")

    def _pred_to_cdr(self, pred):

        return self.cdr[np.argmax(pred)]
        
    def predict_coronal(self, img):
        
        if img.shape != (176,176,3):
            print(f"WARNING: Mismatch in image shape {img.shape} from necessary (176,176,3). Returning None")
            return None

        return self._pred_to_cdr(self.cor_model.predict(np.array([img])))

    def predict_transverse(self, img):
        
        if img.shape != (208,176,3):
            print(f"WARNING: Mismatch in image shape {img.shape} from necessary (208,176,3). Returning None")
            return None

        return self._pred_to_cdr(self.tra_model.predict(np.array([img])))

    def predict_sagittal(self, img):
        
        if img.shape != (176,208,3):
            print(f"WARNING: Mismatch in image shape {img.shape} from necessary (176,208,3). Returning None")
            return None

        return self._pred_to_cdr(self.sag_model.predict(np.array([img])))

    def predict_all(self, cor_img, tra_img, sag_img):
        
        preds = []

        preds.append(self.predict_coronal(cor_img))
        preds.append(self.predict_transverse(tra_img))
        preds.append(self.predict_sagittal(sag_img))

        return preds

class Survey():

    def __init__(self) -> None:
        try:
            print("Loading questions and answers for survey. . .")
            fp = open('Data/conversions.json')
            self.var_dict = json.load(fp)
            self.question_variables = ['_STATE', 'SEXVAR', 'GENHLTH', 'PHYSHLTH', 'MENTHLTH', 'POORHLTH', 'EXERANY2', 'SLEPTIM1', 
                                   'CVDINFR4', 'CVDCRHD4', 'CVDSTRK3', 'ASTHMA3', 'CHCSCNCR', 'CHCOCNCR', 'CHCCOPD2', 'HAVARTH4', 'ADDEPEV3', 
                                   'CHCKDNY2', 'DIABETE4', 'PREGNANT', 'DEAF', 'BLIND', 'DIFFWALK', 'DIFFDRES', 'DIFFALON', 'SMOKE100', 
                                   'USENOW3', 'FALL12MN', 'HIVRISK5', 'TOLDHEPC', 'HAVEHEPB', 'ACEDEPRS', 
                                   'ACEHURT1', 'ACETOUCH', 'ACETTHEM', 'ACEHVSEX', '_RACE', '_AGE_G', '_BMI5CAT', '_RFDRHV7']
        except:
            print("ERROR: Unable to load conversion list for Survey()")
            exit(0)

        
    def _print_choices(self):
        for i, key in enumerate(self.choices):
            mark = ' ' if i+1 != self.active else 'X'

            print(f"[{mark}] {key}: {self.var_dict[self.var][key]}")

    def _update_question_and_choice(self):
        self.var = self.question_variables[self.question_index]
        self.question = self.var_dict[self.var]["?"]
        self.choices = list(self.var_dict[self.var].keys())[1:]
        self._message = None

    def _check_prev_reponse(self):
        if (self.question_index + 1 <= len(self.responses)) and self.responses[self.question_index] != None:
            self.active = self.choices.index(self.responses[self.question_index]) + 1
        else:
            self.active = None

    def _handle_user_input(self):
        if self.userInput.lower() == 'p': # Previous Question
            if self.question_index == 0:
                self._message = "No previous question. You're at the start of the survey."
            else:
                self.question_index -= 1
                self._update_question_and_choice()
                self._check_prev_reponse()
        elif self.userInput.lower() == 'n': # Next Question
            if self.question_index == len(self.question_variables) - 1:
                self._message = "No more questions. Press f to finish survey if you're satisfied with your answers."
            else:

                if len(self.responses) < self.question_index + 1: # Ensure response can be recorded
                    self.responses.append(None)

                if self.active is None:
                    self.responses[self.question_index] = None
                else:
                    self.responses[self.question_index] = self.choices[self.active - 1] # Record response from choice list based on active
                
                self.question_index += 1
                self._update_question_and_choice()
                self._check_prev_reponse()

        elif self.userInput.lower() == 'f': # Finish Survery
            if len(self.responses) != len(self.question_variables):
                print(f"You have only answered {len(self.responses)} out of {len(self.question_variables)} questions.")
                print("Are you sure you want to proceed?")
                userInput = input("(y or n) >> ")

                if userInput.lower() == 'y':
                    self.in_progress = False
            else:
                if None in self.responses:
                    missing_vars = []
                    for i, val in enumerate(self.responses):
                        if val is None:
                            missing_vars.append(self.question_variables[i])
                    
                    print(f"You did not answer questions for {missing_vars}")
                    print("Are you sure you want to proceed?")
                    userInput = input("(y or n) >> ")

                    if userInput.lower() == 'y':
                        self.in_progress = False
                else:
                    self.in_progress = False
        else: # Try selecting a response
            if not (self.userInput in self.choices):
                self._message = (f"{self.userInput} not a valid choice. Please try again")
                return

            self.active = self.choices.index(self.userInput) + 1

            if len(self.responses) < self.question_index + 1: # Ensure response can be recorded
                    self.responses.append(None)

            if self.active is None:
                self.responses[self.question_index] = None
            else:
                self.responses[self.question_index] = self.choices[self.active - 1] # Record response from choice list based on active


    def ask_questions(self):

        # Initialize variables
        self.in_progress = True
        self.question_index = 0
        self.active = None
        self.responses = []

        self._update_question_and_choice()

        # Loop through all the variables and ask the user questions
        while self.in_progress:

            clear_terminal()
            print(f"Variable: {self.var}")
            print(f"{self.question}\n")

            self._print_choices()

            print("\nPlease select a response or press p/n/f to move to previous question, next question, or finish survey")
            
            if not (self.active is None):
                key = self.choices[self.active - 1]
                cur_response = self.var_dict[self.var][key]
            else:
                cur_response = "None"
            if self._message is None:
                self.userInput = input(f" {self.question_index+1}/{len(self.question_variables)} (Current Response: {cur_response}) >> ")
            else:
                self.userInput = input(f"({self._message}) {self.question_index+1}/{len(self.question_variables)} (Current Response: {cur_response}) >> ")
                self._message = None

            # Move to previous question
            self._handle_user_input()

        # Ensure that the length of return responses is same length as questions
        for i in range((len(self.question_variables) - len(self.responses))):
            self.responses.append(None)

        return self.responses

class App():

    def __init__(self) -> None:

        print("Loading necessary program components")      
        self.mri_classifier = MRIClassifier()
        self.bayes_classifier = NaiveBayesianClassifier()
        self.survey = Survey()
        self.current_cd_preds = None
        self.current_cdr_ratings = [None,None,None]
        self.survey_responses = []
        self.excluded_variables = [] # Variables not computed in Bayesian Analysis
        self.callback = "" # Used for callback messages to be printed in menus
        self.subject = None # Keep track of information on the subject ID's for MRI data
        self.return_flag = False # Tracks whether to return to main menu from submenu

        self._load_survey_tests()
        self._load_mri_tests()


        print("All program components loaded")
        print("Press enter key to continue. . .")
        input()

    def _load_survey_tests(self):
        fp = open('Data/survey_tests.txt')
        self.combinations = []
        self.combinations.append(list(fp.readline().split(',')))
        self.combinations[0].pop()
        self.combinations.append(list(fp.readline().split(',')))
        self.combinations[1].pop()
        self.combinations.append(list(fp.readline().split(',')))
        self.combinations[2].pop()
        
        print("Survey tested combinations loaded")

        fp.close()

    def _load_mri_tests(self):
        fp = open('Data/mri_tests.txt')
        self.mri_test_subjects = []
        self.mri_test_subjects.append(fp.readline()[:4])
        self.mri_test_subjects.append(fp.readline()[:4])
        self.mri_test_subjects.append(fp.readline()[:4])
        self.mri_test_subjects.append(fp.readline()[:4])

        print("MRI test subject list loaded")


    def _display_nb_preds(self):

        if self.current_cd_preds is None:
            print("Naive Bayesian Analysis: NO CURRENT RESULTS\n")
        else:
            print(f"Naive Bayesian Analysis: [Yes: {self.current_cd_preds[1]} | No: {self.current_cd_preds[0]} | Maybe: {self.current_cd_preds[2]}]")
            print(f"Prediction of Cognitive Decline: {list(['No','Yes','Maybe/Unknown'])[self.current_cd_preds.index(max(self.current_cd_preds))]}\n")
            if len(self.excluded_variables) > 0:
                print(f"Variables excluded from Bayesian Analysis: {self.excluded_variables}\n")
                print("A variable may have been excluded because either it was left incomplete on the survey or because the BRFSS data did not contain a data instance of the class chosen for that variable\n")

    def _display_mri_preds(self):

        if self.current_cdr_ratings[0] is None and self.current_cdr_ratings[1] is None and self.current_cdr_ratings[2] is None:
            print("MRI Classification: NO CURRENT RESULTS")
            print(f"Subject ID: {self.subject}")
        else:
            print(f"MRI Subject: {self.subject}")
            print(f"MRI Coronal Classification: {self.current_cdr_ratings[0]}")
            print(f"MRI Transverse Classification: {self.current_cdr_ratings[1]}")
            print(f"MRI Sagittal Classification {self.current_cdr_ratings[2]}")
            
            cdr_vals = list(set(self.current_cdr_ratings))
            
            if not (None in cdr_vals) and len(cdr_vals) == 3:
                print(f"CDR rating between {min(cdr_vals)} and {max(cdr_vals)}")
            else:
                most_common = cdr_vals[0]
                for val in cdr_vals:
                    if not (val is None):
                        if self.current_cdr_ratings.count(val) > self.current_cdr_ratings.count(most_common):
                            most_common = val

                if most_common == 0:
                    rating = "Normal"
                elif most_common == 0.5:
                    rating = "Very Mild Dementia"
                elif most_common == 1:
                    rating = "Mild Dementia"
                elif most_common == 2:
                    rating = "Moderate Dementia"
                elif most_common == 3:
                    rating = "Severe Dementia"
                else:
                    rating = "Error"
                print(f"Coalesced CDR Rating: {most_common} - {rating}")
        

    def main_menu(self):

        while True:
            clear_terminal()
            print("\n\n* * * CSC 510: Foundations of Artificial Intelligence * * *")      
            print("  * * * * *      Portfolio Project - Main Menu      * * * * *\n")

            # Display information on both Naive Bayes and MRI Classification
            self._display_nb_preds()
            self._display_mri_preds()

            print("\n(1) Open Survey Screen")
            print("(2) Open MRI Classifier\n")
            
            self.userInput = input(f"(q: quit | r: reset) {self.callback} >> ")
            self.callback = "" # Reset callback after printing it
            self._main_menu_input_handling()

    def _main_menu_input_handling(self):

        if self.userInput == '1':
            print("\nIMPORTANT NOTE: Some variables may be excluded from the bayesian analysis due to selecting a choice which went unselected on the survey or by failing to complete the entire survey.")
            print("\nIn these instances, all excluded variables with be clearly noted along with the probability of cognitive decline correlation.")
            print("\nTo ensure the most accurate results, please complete the survey to its fullest.")
            input("\nPress any key to continue . . .")
            self._survey_menu() # Handle the survey
        elif self.userInput == '2':
            self._mri_menu() # handle MRI Classification
        elif self.userInput.lower() == 'q': # Quit
            exit(0)
        elif self.userInput.lower() == 'r': # Reset
            while self.userInput.lower() != 'y' and self.userInput.lower() != 'n':
                self.userInput = input("Are you sure you want to reset current survey and MRI results? (y or n) >> ")

            if self.userInput.lower() == 'y': # Reset
                self._reset()
        else:
            if self.userInput == "":
                self.callback = ""
            else:
                self.callback = f"[{self.userInput} is not a valid input]" 

    def _survey_menu(self):

        while not self.return_flag:
            clear_terminal()
            print("\n\n* * * CSC 510: Foundations of Artificial Intelligence * * *")      
            print("  * * * * *     Portfolio Project - Survey Menu     * * * * *\n")

            self._display_nb_preds() # Display Niave Bayes info

            print("(1) Run Survey")
            print("(2) Load Precompleted Survey\n")

            self.userInput = input(f"(b: back to main menu | q: quit) {self.callback} >> ")
            self.callback = "" # Reset callback after printing it
            self._survey_menu_input_handling()

        self.return_flag = False

    def _survey_menu_input_handling(self):

        if self.userInput == '1':
            self.survey_responses = self.survey.ask_questions()
            self.current_cd_preds, self.excluded_variables = self.bayes_classifier.predict(self.survey_responses)
            for i,val in enumerate(self.current_cd_preds):
                self.current_cd_preds[i] = round_dec(val, 4)
            # TODO - Display info about excluded variables
        elif self.userInput == '2':
            print("\nPlease choose from the following: ")
            print("(1) Load known probability for NO")
            print("(2) Load known probability for YES")
            print("(3) Load known probability for Maybe")
            self.userInput = ""

            while self.userInput != '1' and self.userInput != '2' and self.userInput != '3':
                self.userInput = input(">> ")
            
            self.survey_responses = self.combinations[int(self.userInput) - 1]
            self.current_cd_preds, self.excluded_variables = self.bayes_classifier.predict(self.survey_responses)
            for i,val in enumerate(self.current_cd_preds):
                self.current_cd_preds[i] = round_dec(val, 4)
            # TODO - Display info about excluded variables
        elif self.userInput.lower() == 'q':
            exit(0)
        elif self.userInput.lower() == 'b':
            self.return_flag = True
        else:
            if self.userInput == "":
                self.callback = ""
            else:
                self.callback = f"[{self.userInput} is not a valid input]" 

    def _mri_menu(self):

        if self.subject == None:
            self._choose_subject()

        while not self.return_flag:
            clear_terminal()
            print("\n\n* * * CSC 510: Foundations of Artificial Intelligence * * *")      
            print("  * * * * *      Portfolio Project - MRI Menu       * * * * *\n")

            self._display_mri_preds() # Display Niave Bayes info

            print("\n(1) Process Full Subject")
            print("(2) Process Coronal MRI")
            print("(3) Process Transverse MRI")
            print("(4) Process Sagittal MRI")
            print("(5) Load Test Images")
            print("(6) Choose New Subject\n")

            self.userInput = input(f"(b: back to main menu | q: quit) {self.callback} >> ")
            self.callback = "" # Reset callback after printing it
            self._mri_menu_input_handling()

        self.return_flag = False

    def _mri_menu_input_handling(self):
        
        if self.userInput == '1':
            # Load all images
            print("Loading all subject MRI images")
            cor_image = self._load_subject_img('coronal')
            tra_image = self._load_subject_img('transverse')
            sag_image = self._load_subject_img('sagittal')

            # Processing all iamges
            print("Processing all subject MRI images")
            self.current_cdr_ratings = self.mri_classifier.predict_all(cor_image,tra_image,sag_image)
            self._plot_all_scans(cor_image, tra_image, sag_image)
        
        elif self.userInput == '2':
            print(f"\nLoading coronal image for subject {self.subject}")
            image = self._load_subject_img('coronal')
            print("\nProcessing coronal image")
            self.current_cdr_ratings[0] = self.mri_classifier.predict_coronal(image)
            self._plot_single_scans(image, 'coronal')
        
        elif self.userInput == '3':
            print(f"\nLoading transverse image for subject {self.subject}")
            image = self._load_subject_img('transverse')
            print("\nProcessing transverse image")
            self.current_cdr_ratings[1] = self.mri_classifier.predict_transverse(image)
            self._plot_single_scans(image, 'transverse')
        
        elif self.userInput == '4':
            print(f"\nLoading sagittal image for subject {self.subject}")
            image = self._load_subject_img('sagittal')
            print("\nProcessing sagittal image")
            self.current_cdr_ratings[2] = self.mri_classifier.predict_sagittal(image)
            self._plot_single_scans(image, 'sagittal')
        
        elif self.userInput == '5':
            # TODO - Put functionality here
            print("\nPlease choose from one of the following: ")
            print("(1) Load known 0.0 CDR Score MRI")
            print("(2) Load known 0.5 CDR Score MRI")
            print("(3) Load known 1.0 CDR Score MRI")
            print("(4) Load known 2.0 CDR Score MRI")
            self.userInput = ""

            while not (self.userInput in ['1','2','3','4']):
                self.userInput = input(">> ")

            self.subject = self.mri_test_subjects[int(self.userInput) - 1]

            print("Loading all subject MRI images")
            cor_image = self._load_subject_img('coronal')
            tra_image = self._load_subject_img('transverse')
            sag_image = self._load_subject_img('sagittal')

            # Processing all iamges
            print("Processing all subject MRI images")
            self.current_cdr_ratings = self.mri_classifier.predict_all(cor_image,tra_image,sag_image)
            self._plot_all_scans(cor_image, tra_image, sag_image)
            
        
        elif self.userInput == '6':
            while self.userInput.lower() != 'y' and self.userInput.lower() != 'n':
                self.userInput = input("Please confirm decision to change subjects and reset any current classification results? (y or n) >> ")
            if self.userInput.lower() == 'y':
                self._choose_subject()
                self.current_cdr_ratings = [None,None,None]
        
        elif self.userInput.lower() == 'q':
            exit(0)
        
        elif self.userInput.lower() == 'b':
            self.return_flag = True
        
        else:
            if self.userInput == "":
                self.callback = ""
            else:
                self.callback = f"[{self.userInput} is not a valid input]" 
        

    def _choose_subject(self):
        self.subject = None
        self.active = None
        subjects = []
        for file in os.listdir("Data/MRI/coronal/"):
            subject_id = file[5:9]
            if subject_id != 'tore':
                subjects.append(subject_id)

        subjects.sort()
        while self.subject is None:

            clear_terminal()
            print("\n\n* * * CSC 510: Foundations of Artificial Intelligence * * *")      
            print("  * * * * *      Portfolio Project - MRI Menu       * * * * *\n")
            print("NO SUBJECT DETECTED. Please choose from following list:\n")

            newline = 0
            for i, name in enumerate(subjects):
                
                if i+1 < 10:
                    ind = f"( {i+1} )"
                elif i+1 < 100:
                    ind = f"( {i+1})"
                else: 
                    ind = f"({i+1})"

                mark = ' ' if i+1 != self.active else 'X'
                print(f"[{mark}] {ind}: Subject {name}", end='')
                newline += 1

                if newline == 4:
                    print("")
                    newline = 0
                else:
                    print("   ", end='')
            
            print('\n')

            if self.active != None:
                self.userInput = input(f"Please choose a subject (c: confirm | Current Subject ID: {subjects[self.active - 1]}) {self.callback}>> ")
            else:
                self.userInput = input(f"Please choose a subject (c: confirm | Current Subject ID: None) {self.callback}>> ")

            self.callback = "" # Reset callback after printing

            if self.userInput == 'c':
                if self.active == None:
                    self.callback = "[Please choose a subject]"
                else:
                    self.subject = subjects[self.active - 1]
            else:
                try:
                    n = int(self.userInput)
                    if n > 0 and n < len(subjects) + 1:
                        self.active = n
                    else:
                        self.callback =  f"[{self.userInput} is not a valid input]"
                except:
                    self.callback =  f"[{self.userInput} is not a valid input]"
            
    def _load_subject_img(self,folder):
        '''Loads subject mri image from folder (coronal/transverse/sagittal) and returns it'''

        img = None
        n3_substr = "_mpr_n3_anon_111_t88_gfc_"
        n4_substr = "_mpr_n4_anon_111_t88_gfc_"
        n5_substr = "_mpr_n5_anon_111_t88_gfc_"
        n6_substr = "_mpr_n6_anon_111_t88_gfc_"

        substr_options = [n3_substr,n4_substr,n5_substr,n6_substr]

        if folder == 'coronal':
            tag = "cor_110.jpg"
        elif folder == 'transverse':
            tag = "tra_90.jpg"
        elif folder == 'sagittal':
            tag = "sag_95.jpg"
        else:
            print("ERROR: Unrecognized folder param in image load")
            exit()

        for sub_op in substr_options:
            if not (img is None):
                break
            try:
                path = os.path.join("Data/MRI", folder, f"OAS1_{self.subject}_MR1" + sub_op + tag)
                img = cv.imread(path)
            except:
                pass

        return img

    def _reset(self):
        self.current_cd_preds = None
        self.current_cdr_ratings = [None,None,None]
        self.subject = None
        self.callback = "[data reset]"
        self.return_flag = False
        self.excluded_variables = []

    def _plot_single_scans(self, image, orientation):

        fig, axs = plt.subplots(1)

        if orientation == 'coronal':
            cdr = self.current_cdr_ratings[0]
        elif orientation == 'transverse':
            cdr = self.current_cdr_ratings[1]
        elif orientation == 'sagittal':
            cdr = self.current_cdr_ratings[2]

        fig.suptitle(f"{orientation} scan for subject {self.subject} - Estimated CDR Rating: {cdr}")
        
        axs.set_xticks([])
        axs.set_yticks([])

        axs.imshow(image)

        plt.show()

    def _plot_all_scans(self,cor_img, tra_img, sag_img):
        fig, axs = plt.subplots(1,3)

        cdr_vals = list(set(self.current_cdr_ratings))
            
        if not (None in cdr_vals) and len(cdr_vals) == 3:
            print(f"CDR rating between {min(cdr_vals)} and {max(cdr_vals)}")
        else:
            most_common = cdr_vals[0]
            for val in cdr_vals:
                if not (val is None):
                    if self.current_cdr_ratings.count(val) > self.current_cdr_ratings.count(most_common):
                        most_common = val

            if most_common == 0:
                rating = "Normal"
            elif most_common == 0.5:
                rating = "Very Mild Dimentia"
            elif most_common == 1:
                rating = "Mild Dimentia"
            elif most_common == 2:
                rating = "Moderate Dimentia"
            elif most_common == 3:
                rating = "Severe Dimentia"
            else:
                rating = "Error"

        fig.suptitle(f'MRI scans for subject {self.subject} - CDR Rating ({most_common}): {rating}')

        for i in range(3):
                axs[i].set_xticks([])
                axs[i].set_yticks([])

        axs[0].set_title(f'Coronal ({self.current_cdr_ratings[0]})')
        axs[1].set_title(f'Transverse ({self.current_cdr_ratings[1]})')
        axs[2].set_title(f'Sagittal ({self.current_cdr_ratings[2]})')

        axs[0].imshow(cor_img)
        axs[1].imshow(tra_img)
        axs[2].imshow(sag_img)

        plt.show()

if __name__ == "__main__":
    app = App()

    app.main_menu()