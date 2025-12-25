# Customer Segmentation & Purchase Prediction

## 🚀 Project Overview
This project combines **Unsupervised** and **Supervised** machine learning techniques to analyze customer data and provide business insights.  

- **Unsupervised Learning (KMeans Clustering):** Group customers into distinct segments based on demographics and spending behavior.  
- **Supervised Learning (Logistic Regression / Random Forest):** Predict whether a customer will make a purchase or not.

**Objective:**  
1. Identify customer segments for targeted marketing.  
2. Predict customer purchase behavior to improve sales strategy.

---

## 📁 Dataset
- Dataset contains **5000+ customers** with the following features:
  | Column | Description |
  |--------|------------|
  | CustomerID | Unique customer identifier |
  | Age | Customer age |
  | Gender | Male / Female |
  | City | Customer city |
  | Annual_Income | Yearly income in PKR |
  | Spending_Score | Spending score (1–100) |
  | Visit_Frequency | Number of visits per month |
  | Online_Shopping | 0 = No, 1 = Yes |
  | Purchase | Target: 0 = No, 1 = Yes |

- [Download Dataset](sandbox:/mnt/data/heavy_customer_segmentation_dataset.csv)

---

## 🧩 Tools & Libraries
- Python 3.x  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn (StandardScaler, KMeans, PCA, LogisticRegression, RandomForest)  

---

## 🔍 Workflow

1. **Data Cleaning:** Handle missing values, duplicates, and correct data types.  
2. **Exploratory Data Analysis (EDA):** Analyze distributions, correlations, and visualize patterns.  
3. **Feature Scaling:** Standardize numerical features for clustering.  
4. **Customer Segmentation (KMeans):** Identify customer clusters based on Age, Income, Spending Score, and Visit Frequency.  
5. **PCA Visualization:** Reduce dimensions to 2D for cluster visualization.  
6. **Purchase Prediction:** Train supervised models to predict if a customer will make a purchase.  
7. **Model Evaluation:** Use accuracy, precision, recall, and confusion matrix to evaluate performance.  
8. **Business Insights:** Generate actionable insights for each customer segment.  

---

## 📊 Sample Business Insights

| Cluster | Avg Purchase Rate | Insight |
|---------|-----------------|---------|
| 0       | 0.38             |Average Customer |
| 1       | 0.75            | High income + high spending → Premium Customer |
| 2       | 0.22            | Low spending → loyalty program |

**Visualization:**  
- Bar charts showing **purchase rate per cluster**  
- PCA scatter plot showing cluster separation

---

## 💡 Key Takeaways
- Combining **unsupervised + supervised learning** provides both **segmentation and prediction**.  
- Actionable **Business Insights** help marketing and sales teams target the right customers.  
- Dataset features like **Age, Income, Spending Score, Visit Frequency** are crucial for accurate segmentation and prediction.

---

## 📂 Project Structure
- ├── heavy_customer_segmentation_dataset.csv
- ├── heavy_customer_segmentation_dataset.py
- ├── README.md

---

## 🔗 References
- [KMeans Clustering - Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)  
- [PCA - Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)  
- [Logistic Regression - Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)  

---

## 🚀 How to Run the Project
1. Clone the repository:

   git clone https://github.com/syedhassannaseem/Customer-Segmentation-Purchase-Prediction
