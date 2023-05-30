# Personal Professional Project :
#### Predicting Duplicate Questions in a question-and-answer platform

![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

### Order of execution of the notebooks :

1. #### Exploratory Data Analysis :
     ###### explore the dataset , identify its properties and outliers and generate visualizations
   * questions_analysis.ipynb
   * data_cleaning.ipynb
   * visualization.ipynb
   
2. #### Feature Engineering :
      ###### transforme raw text into a format that is suitable for the Model using NLP techniques
    * nlp_preprocessing.ipynb
    * nlp_processing.ipynb
    * processed_preparing.ipynb
   
3. #### Models Training :
      ###### develop and train deep learning models using the prepared dataset as well as evaluate them
    * data_splitting.ipynb
    * model_building.ipynb
    * model_testing.ipynb

This project addresses the problem of predicting duplicate questions in question-answering systems. The aim is to develop an effective deep learning model capable of accurately identifying redundant queries, thereby improving search efficiency and user experience

This is a group project for the course **Professional Personal Project** at the **National Institute of Applied Science and Technology**, Tunisia.

## Folders

The project consists of the following folders:

- [config](/config): Contains some necessary configuration files like init.py that appends 'src' directory to the system path.

- [data](/data): Stores the dataset and its variations throughout the whole span of the project to avoid redoing data transformation processes and simply load them whenever needed.

- [models](/models): Stores trained models versions.
   
- [notebooks](/notebooks): houses Jupyter notebooks used for our different processes ( View Order of execution on the top ).
   
- [reports](/reports): holds generated reports, such as the model graph.

- [src](/src): Contains the scripts of the functions used in the notebooks to promote code organization and maintainability.

## Getting Started

To run the project, follow the steps below:

1. clone the repository by using the following command:

    ```bash
    git clone https://github.com/Dhouib-Mohamed/Duplicate-Question-Predictor
    ```

2. Install the required packages listed in [requirements.txt](requirements.txt) using the following command:
    
    ```bash
    pip install -r requirements.txt
    ```
3. Run the necessary configuration in config file:

     ```bash
     python .\config\__init__.py
     ```

4. Run Each Notebook in the correct order

## Project overview: Methodology and Approach

### Data Pre-processing

The data pre-processing step includes the following steps:
- **Case Normalization**: Convert all text to lowercase.
- **Data Cleaning**: Remove special characters, and ponctuation.
- **Stopwords Removal**: Remove stopwords from the text.
- **Lemmalization**: Extracting the lemma from each word.

### Feature Engineering

The feature engineering step includes the following steps:
- **Gensim Vectorization**: Convert text to a matrix of Gensim features.

### Model Training and Evaluation

The model training and evaluation step includes the following steps:
- **Train/Test Split**: Split the data into training and testing sets.
- **Model Training**: Train a classifier model using the training set.
- **Model Evaluation**: Evaluate the model using the testing set.
  
accuracy:   0.68513
              precision    recall  f1-score   support

    Positive       0.72      0.82      0.76     45989
    Negative       0.61      0.47      0.53     28090

    accuracy                           0.69     74079
   macro avg       0.66      0.64      0.65     74079
weighted avg       0.68      0.69      0.67     74079


## Additional Notes

- The project's code is provided in the Jupyter notebook [step-by-step-guide.ipynb](/Step-by-step-guide.ipynb), which contains detailed explanations and code snippets for each step.

- The project utilizes various Python packages such as pandas, NLTK, scikit-learn, Matplotlib, seaborn, keras... . Make sure to install these packages, as mentioned in the requirements.txt file.

## Team Members
- [Dhouib Mohamed](https://github.com/Dhouib-Mohamed)
- [Nada Mankai](https://github.com/nadamankai)
- [Farah Seddik](https://github.com/farahsedd)
