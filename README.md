#  Heart Failure Prediction

This project explores classification as a form of supervised machine learning, using R to predict heart failure risk based on real-world patient data.

## 📂 Project Structure

- `data/` – contains the dataset (`heart.csv`)
- `knn/` – K-Nearest Neighbors implementation
- `naive_bayes/` – Naive Bayes Classifier implementation
- `decision_tree/` – Decision Tree implementation

Each folder includes a single R script that handles the complete pipeline.



## ⚙️ Process Performed in Each Script

- **Data Preparation and Cleaning**  
- **Exploratory Data Analysis (EDA)**   
- **Model Training**   
- **Model Evaluation** 
- **Visualization** 



## 📈 Dataset

- Source: [Kaggle - Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)  
- Format: CSV  
- Location: `data/heart.csv`  

---

## 🧪 How to Run

Open a desired .R script in **RStudio** and run the code.
Make sure the dataset file heart.csv is located in the same directory as the script.
If the dataset is in a different location, update the path accordingly.

Example of reading the dataset from the script:

```r
data <- read.csv("heart.csv", stringsAsFactors = FALSE)
 ``` 
---

📊 For a full comparison of all models, see the [Model Comparison Summary](Model_comparison.md).
