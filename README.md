# Deep Learning Homework: Charity Funding Predictor
![Deep_Learning](/Images/Deep_Learning2.jpg)

## Background

The non-profit foundation Alphabet Soup wants to create an algorithm to predict whether or not applicants for funding will be successful. With my knowledge of machine learning and neural networks, I will be using the features in the provided dataset to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, I have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

* **EIN** and **NAME**—Identification columns
* **APPLICATION_TYPE**—Alphabet Soup application type
* **AFFILIATION**—Affiliated sector of industry
* **CLASSIFICATION**—Government organization classification
* **USE_CASE**—Use case for funding
* **ORGANIZATION**—Organization type
* **STATUS**—Active status
* **INCOME_AMT**—Income classification
* **SPECIAL_CONSIDERATIONS**—Special consideration for application
* **ASK_AMT**—Funding amount requested
* **IS_SUCCESSFUL**—Was the money used effectively

### Below are the steps I took to get this project executed.

### Step 1: Preprocess the data

Preprocessing is very important and considered the initial step in machine learning. Within the preprocessing, the dataset is cleaned by dropping the noise columns that are not beneficial to the machine learning model. Using my knowledge of Pandas and the Scikit-Learn’s `StandardScaler()`, I preprocessed the dataset in order to compile, train, and evaluate the neural network model later in Step 2.

To complete the preprocessing of the dataset, I read in the charity_data.csv to a Pandas DataFrame. It is very important to identify the target variables and the feature variables for the model.
  * What variable(s) are considered the target(s) for your model?
   * In this model, the target variables are shown as the "IS_SUCCESSFUL" column. This will be the desired outcome of the model. 
  
  * What variable(s) are considered the feature(s) for your model?
   * In this model, the feature varibales are all the columns in the dataset except the `EIN` and `NAME` columns since these two columns were dropped from the initial preprocessing stage.
   
![Preprocessor](/Images/Preprocessor.jpg)

The number of unique values for each column were determined and the for those columns that have more than 10 unique values, the number of data points for each unique value were determined as well. I used the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then I checked if the binning was successful. I then used `pd.get_dummies()` to encode categorical variables.

### Step 2: Compile, Train, and Evaluate the Model

In step 2, using my knowledge of TensorFlow, I designed a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. I thought about how many inputs there are before determining the number of neurons and layers in your model. Once I completed that step, I compiled, trained, and evaluated my binary classification model to calculate the model’s loss and accuracy.

![Model_Setup](/Images/Model_Setup.jpg)

### Step 3: Optimize the Model

In step 3, I used my knowledge of TensorFlow to optimize my model in order to achieve a targeted predictive accuracy higher than 75%. During the optimization of the model I had to do some trial and error of:
* Adjusting the input data to ensure that there are no variables or outliers that are causing confusion in the model, such as:
  * Dropping more or fewer columns.
  * Creating more bins for rare occurrences in columns.
  * Increasing or decreasing the number of values for each bin.
* Adding more neurons to a hidden layer.
* Adding more hidden layers.
* Using different activation functions for the hidden layers.
* Adding or reducing the number of epochs to the training regimen.

One interesting observation was that with the `EIN` and `NAME` columns  dropped earlier in the preprocessing stage, the model achieved 72.6% accuracy with the initial run. All the additional runs were not providing accuracy higher than 75%. After reviewing the dataset one more time, I decided to undrop the `NAME`column and then continue with the optimization process. When the `NAME` column was undropped, I did achieve an accuracy of 78.8% which is higher than the original anticipated 75% accuracy. At this point it was really clear that the `NAME` colum played a pivotal role in properly increasing the accuracy of the machine learning model. I decided to run additional runs with introducing some additional hiden layer, changing the activation and also the units per layer. I did also observed that a combination of relu and sigmoid activations provided the best results than the other actions.

### Step 4: Write a Report on the Neural Network Model

Below is a report on the performance of the deep learning model that I created for AlphabetSoup.

1. **Overview of the analysis**: 
    The main purpose of this machine learning model is to help predict the success level of an applicant for charity funding with the non-profit foundation Alphabet Soup.

2. **Results**: 

  * Data Preprocessing
    * What variable(s) are considered the target(s) for your model?
      In this model, the target variables are shown as the "IS_SUCCESSFUL" column. This will be the desired outcome of the model. 

    * What variable(s) are considered the feature(s) for your model?
      In this model, the feature varibales are all the columns in the dataset except the `EIN` and `NAME` columns since these two columns were dropped from the initial preprocessing stage.
    * What variable(s) are neither targets nor features, and should be removed from the input data?
      Since the `NAME` column was undropped for the optimization of the model, there was only one variable that was considered as neither a target nor a feature and that was the `EIN` column for the model adopted in this machine learning project.
    
  * Compiling, Training, and Evaluating the Model
    * How many neurons, layers, and activation functions did you select for your neural network model, and why?
      In the initial run, I used 1 input layer, 1 hidden layer, and 1 output layer set at 80, 30, 1 neurons respectively with relu activation for model input and hidden layer and then sigmoid activation for the output layer. In the second run with was the first optimization run with the highest accuracy,I used 1 input layer, 1 hidden layer, and 1 output layer set at 80, 30, 1 neurons respectively with relu activation for model input, sigmoid activation for hidden layer and then sigmoid activation for the output layer. I then run couple additional optimization models and the results are provided below
    * Were you able to achieve the target model performance?
      Yes, I was able to achieve and surpass the target model performance of 75% reaching 78.8% accuracy
      
    * What steps did you take to try and increase model performance?
      In the second run with was the first optimization run with the highest accuracy,I used 1 input layer, 1 hidden layer, and 1 output layer set at 80, 30, 1 neurons respectively with relu activation for model input, sigmoid activation for hidden layer and then sigmoid activation for the output layer. It is very important to also note that in order for me to achieve the increased model performance of 78.8%, I had to include the `NAME` column in my model and not drop it. It did play significant role in determining the accuracy level of the model.
      
3. **Summary**:
The machine learning model is very intriguing. It does however require a series of trial and error with different neurons and activations to achieve a targeted accuracy level. In this projected, it was very interesting to see how important certain features are in the determination of higher accuracy level of the machine learning model. In this case, `NAME` was crucial and cannot be secluded in the model. Even with the inclusion of `NAME` as a feature. Below are the various figures of the loss and accuracy recorded during the project.
---
**Loss and Accuracy graphs**

![Initial_Run](/Images/Initial_Run.jpg)      
![Accuracy Plot 1](/Images/accuracy_plot1.PNG) ![Loss Plot 1](/Images/loss_plot1.PNG)

---
![Optimization 1](/Images/Optimization_1.jpg)      
![Accuracy Plot 2](/Images/accuracy_plot2.PNG) ![Loss Plot 2](/Images/loss_plot2.PNG)

---
![Optimization 2](/Images/Optimization_2.jpg)      
![Accuracy Plot 3](/Images/accuracy_plot3.PNG) ![Loss Plot 1](/Images/loss_plot3.PNG)

---
![Optimization 3](/Images/Optimization_3.jpg)      
![Accuracy Plot 4](/Images/accuracy_plot4.PNG) ![Loss Plot 1](/Images/loss_plot4.PNG)

---
