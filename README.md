# FlipItNews NLP Classification — News Article Categorization

Domain: Natural Language Processing (NLP) • Tech Stack: Python | scikit-learn | NLTK | Pandas | TF-IDF | Bag of Words | Machine Learning

## Project Overview

  - The project focuses on categorizing news articles from the Indian financial, business, and investment domain into multiple categories like Politics, Technology, Sports, Business, and Entertainment.
  
  - Inspired by FlipItNews, a Gurugram-based company aiming to enhance financial literacy through AI, this project demonstrates end-to-end NLP workflow — from text preprocessing to vectorization, model training, and evaluation.
  
  - It is designed as a portfolio project for showcasing NLP and machine learning skills.

## Dataset

- The dataset consists of news articles and their categories:
  | Attribute | Description |
  |:----------|:------------|
  | Article   | The text of the news article |
  | Category  | Target label (Politics, Technology, Sports, Business, Entertainment) |


- Dataset Link: https://drive.google.com/file/d/1I3-pQFzbSufhpMrUKAROBLGULXcWiB9u/view

## Concepts Practiced
  
  - Text Preprocessing: Removing non-letters, stopwords, tokenization, lemmatization
  
  - Text Vectorization: Bag of Words, TF-IDF
  
  - Multi-class Classification using ML models
  
  - Train-Test split and data transformation
  
  - Evaluation: Accuracy, Confusion Matrix, Classification Report
  
  - Functionalizing code to train multiple classifiers

## Project Workflow
  - #### 1. Data Loading & Exploration
  
    - Installed and imported required libraries: pandas, scikit-learn, nltk, matplotlib, seaborn.
    
    - Loaded dataset and examined its structure.
    
    - Checked distribution of articles across categories.
  
 - #### 2. Text Preprocessing
  
    - Removed non-letter characters.
    
    - Tokenized the text.
    
    - Removed stopwords.
    
    - Performed lemmatization.
    
    - Displayed examples of articles before and after preprocessing.
  
  - #### 3. Encoding & Vectorization
  
    - Encoded the target variable (Category) using LabelEncoder.
    
    - Vectorized the text using:
    
    - Bag of Words
    
    - TF-IDF (user-selectable option)
    
    - Split dataset into train (75%) and test (25%) sets.
  
  - #### 4. Model Training & Evaluation
  
    - Trained and compared multiple classifiers:
      
      | Model | Description |
      |:------------------|:--------------------------------------|
      | Naive Bayes        | Baseline probabilistic model           |
      | Decision Tree      | Tree-based classifier for interpretability |
      | Nearest Neighbors  | Instance-based classifier              |
      | Random Forest      | Ensemble of Decision Trees             |

    - Evaluated models using:
    
      - Accuracy
      
      - Confusion Matrix
      
      - Classification Report (Precision, Recall, F1-Score)
    
    - Observed and commented on relative performance of each model.

## Key Results & Insights

  - Number of articles: (Insert number)
  
  - Most common category: (Insert category)
  
  - Technology articles: (Insert number)
  
  - Best performing model: Random Forest (based on accuracy, precision, recall)
  
  - Observation: Both precision and recall are equally important for this use case.
  
  - Efficiency of vectorization techniques: TF-IDF is generally more informative than Bag of Words.
  
  - Train-Test Shape: 75:25 split ensures sufficient data for training and evaluation.

## Tools & Libraries
  | Category | Tools / Libraries |
  |:----------------------|:-----------------------------|
  | NLP & Text Processing  | NLTK, re, string            |
  | Data Manipulation      | Pandas, NumPy               |
  | Machine Learning       | scikit-learn                |
  | Visualization          | Matplotlib, Seaborn         |

## Key Learnings

  - Learned the complete NLP workflow from raw text to ML model evaluation.
  
  - Understood text cleaning, tokenization, stopword removal, and lemmatization.
  
  - Compared classical ML models for multi-class text classification.
  
  - Explored vectorization techniques (Bag of Words vs TF-IDF) and their impact.
