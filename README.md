# CSC 510 - Foundations of Artificial Intelligence
**Disclaimer:** These projects were built as a requirement for CSC 510: Foundations of Artificial Intelligence at Colorado State University Global under the instruction of Dr. Jonathan Vanover. Unless otherwise noted, all programs were created to adhere to explicit guidelines as outlined in the assignment requirements I was given. Descriptions of each [programming assignment](#programming-assignments) and the [portfolio project](#portfolio-project) can be found below.

*****This class has been completed, so this repository is archived.*****
___

### Languages and Tools
[<img align="left" height="32" width="32" src="https://cdn.svgporn.com/logos/python.svg" />](https://www.python.org)
[<img align="left" height="32" width="32" src="https://www.psych.mcgill.ca/labs/mogillab/anaconda2/lib/python2.7/site-packages/anaconda_navigator/static/images/anaconda-icon-512x512.png" />](https://www.anaconda.com/pricing)
[<img align="left" height="32" width="32" src="https://cdn.svgporn.com/logos/visual-studio-code.svg" />](https://code.visualstudio.com)
[<img align="left" height="32" width="32" src="https://cdn.svgporn.com/logos/git-icon.svg" />](https://git-scm.com)
[<img align="left" height="32" width="32" src="https://cdn.svgporn.com/logos/gitkraken.svg" />](https://www.gitkraken.com)
[<img align="left" height="32" width="32" src="https://cdn.svgporn.com/logos/tensorflow.svg" />](https://www.tensorflow.org)
[<img align="left" height="32" width="32" src="https://cdn.svgporn.com/logos/jupyter.svg" />](https://jupyter.org)
<br />

### Textbook
The textbook for this class was [**Operating Systems: Internals and Design Principles**](https://www.pearson.com/us/higher-education/program/Stallings-Operating-Systems-Internals-and-Design-Principles-9th-Edition/PGM1262980.html) by **William Stallings**
### VS Code Comment Anchors Extension
I am also using the [Comment Anchors extension](https://marketplace.visualstudio.com/items?itemName=ExodiusStudios.comment-anchors) for Visual Studio Code which places anchors within comments to allow for easy navigation and the ability to track TODO's, code reviews, etc. You may find the following tags intersperesed throughout the code in this repository: ANCHOR, TODO, FIXME, STUB, NOTE, REVIEW, SECTION, LINK, CELL, FUNCTION, CLASS

For anyone using this extension, please note that CELL, FUNCTION, and CLASS are tags I defined myself. 
<br />

___
<!--When doing relative paths, if a file or dir name has a space, use %20 in place of the space-->
## Programming Assignments
### Critical Thinking 3: [Tensorflow ANN Model](CT%203/)
- A basic Tensorflow ANN model trained to model a simple mathematical function.

### Critical Thinking 4: [Informed Search Heuristics with SimpleAI](CT%204)
- This program utilizes the [Simple AI](https://github.com/simpleai-team/simpleai) library to implement an informed heuristic solution to a real-world inspired search problem. I defined a theoretical based on the premise of the vacuum problem presented by Norvig and Russel in [**Artificial Intelligence: A Modern Approach**](https://www.pearson.com/us/higher-education/program/Russell-Artificial-Intelligence-A-Modern-Approach-3rd-Edition/PGM156683.html). The scenario for the program was based on the idea of a robot vacuum that would need to find the optimal path to vacuum up some number of dirt piled and return to its starting location. 
    - The A* search algorithm was utilized as the informed search algorithm
    - Each dirt pile was treated as a waypoint and at after each action a new optimal path was calculated
        - The program was not computationally efficient since no waypoint paths were cached and recalculation occured at every step in the program

### Critical Thinking 6: [Naïve Bayes Classifier](CT%206)
- This program explored the implementation of a Naïve Bayes classifier as implemented with [scikit-learn](https://scikit-learn.org/stable/) The [Iris Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) provided with scikit-learn was used for the implementation of the Bayesian classifier. Two different Bayesian classifier were ultimately implemented:
    1. Naïve Bayes classifier with categorical data. The Iris Dataset includes continuous numerical data, so a categorical representation for each feature was generated and copy of the dataset was created with these conversions implemented. Pandas was used to create and visualize the likelihood tables during prediction.
    2. Gaussian Naïve Bayes classifier with original numerical data. Scikit-learns `GaussianNB()` class was trained on the Iris Dataset and used for interactive prediciton based on user inputted characteristics. 
___
## Portfolio Project [Cognitive Decline Predictor](Portfolio%20Project)
- This project was an attempt at a rudimentary program for detecting and classifying cognitive decline in individuals based on survey responses and MRI data. The project was created as a class assignment with a time frame lasting only a few weeks. It was used to explore a few AI methods, demonstrate my ability to implement the various methods, and to demonstrate an ability to create a **working AI program**. No claims are made as to the accuracy or validity of the program's inferences and robust testing for accuracy was not done. 
- This project is divided into two parts:
    1. Naive Bayesian analysis of a behavior questionnaire
    2. CNN classification of MRI scans
- The general idea is that a *patient* would answer a series of questions which would be analyzed with Naive Bayesian Analysis. The analysis would indicate how likely the *patient* were to have a symptom of cognitive decline. Based on this response, a *physician* could decide to analyze some MRI images to get an estimated CDR rating. 
#### 1. Naive Bayesian Analysis
- Data was acquired from the Centers for Disease Control and Prevention's [2020 BRFSS Survey Data and Documentation](https://www.cdc.gov/brfss/annual_data/annual_2020.html)
- The BRFSS survey data contains a number of questions under a grouping of *Cognitive Decline*
    - All responses to cognitive decline were coalesced into a single Y/N variable
    - 41 variables were chosen (related to Age, Ethinicity, and various Health Factors) for consideration
    - Any responses that were missing data for these 41 variables were filtered out
- Naive Bayesian Analysis was chosen for ease of use since each variable is considered indepentently of one another
- The 41 variables and the associated questions and responses from the 2020 BRFSS Survey are presented to a user when running the program. The results analyze the responses to return a probabilty (Yes/No/Unknown) on whether the given responses would indicate answering Yes on any of the cognitive decline questions

#### 2. CNN Classification of MRI Scans
- Acknowledgement: Data were provided by OASIS: Cross-Sectional: Principal Investigators: D. Marcus, R, Buckner, J, Csernansky J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382
    - OASIS: Cross-Sectional: https://doi.org/10.1162/jocn.2007.19.9.1498 
- The OASIS data listed a [Cognitive Decline Rating (CDR)](https://knightadrc.wustl.edu/cdr/cdr.htm) associated with each patient
    - Any records missing a CDR rating were filtered out
- For each record with a CDR rating, the coronal, transverse, and sagittal MRI's were filtered out
    - Three CNNs were trained on these MRI images to try and classify then with a CDR rating: One for each imaging plane
    - The collective CDR rating was taken as a combination of the three CNN outputs
- **NOTE:** The MRI section in the Data folder does not currenlty contain images since they were provided from the OASIS data set. The subfolders under the MRI section are where the MRI images corresponding to each plane would go