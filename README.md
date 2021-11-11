# CSC 510 - Foundations of Artificial Intelligence
**Disclaimer:** These projects were built as a requirement for CSC 510: Foundations of Artificial Intelligence at Colorado State University Global under the instruction of Dr. Jonathan Vanover. Unless otherwise noted, all programs were created to adhere to explicit guidelines as outlined in the assignment requirements I was given. Descriptions of each [programming assignment](#programming-assignments) and the [portfolio project](#portfolio-project) can be found below.

*****This class has been completed, so this repository is archived.*****
___

### Languages and Tools
<!--TODO add links to each icons site -->
<img align="left" height="32" width="32" src="https://cdn.svgporn.com/logos/python.svg" />
<img align="left" height="32" width="32" src="https://cdn.svgporn.com/logos/github-octocat.svg" />
<img align="left" height="32" width="32" src="https://www.psych.mcgill.ca/labs/mogillab/anaconda2/lib/python2.7/site-packages/anaconda_navigator/static/images/anaconda-icon-512x512.png" />
<img align="left" height="32" width="32" src="https://cdn.svgporn.com/logos/visual-studio-code.svg" />
<img align="left" height="32" width="32" src="https://cdn.svgporn.com/logos/git-icon.svg" />
<img align="left" height="32" width="32" src="https://cdn.svgporn.com/logos/gitkraken.svg" />
<img align="left" height="32" width="32" src="https://cdn.svgporn.com/logos/tensorflow.svg" />
<br />

### Textbook
The textbook for this class was [**Operating Systems: Internals and Design Principles**](https://www.pearson.com/us/higher-education/program/Stallings-Operating-Systems-Internals-and-Design-Principles-9th-Edition/PGM1262980.html) by **William Stallings**

### VS Code Comment Anchors Extension
I am utilizing the [Comment Anchors extension](https://marketplace.visualstudio.com/items?itemName=ExodiusStudios.comment-anchors) for Visual Studio Code which places anchors within comments to allow for easy navigation and the ability to track TODO's, code reviews, etc.
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
## Portfolio Project
[INFORMATION COMING SOON - I know the repo is archived, but I do intended on filling this information in any day now]
