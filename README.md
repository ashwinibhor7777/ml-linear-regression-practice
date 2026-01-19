# ml-linear-regression-practice


# Building and Evaluating a Linear Regression Model Using Machine Learning

## 📌 Project Overview

This project demonstrates an end-to-end **Supervised Machine Learning** workflow using **Linear Regression** to build a prediction model. The focus is on applying proper **data preprocessing, feature engineering, model training, evaluation, and performance analysis** using real-world–style data.

The project is implemented in **JupyterLab** using **Python** and popular ML libraries.

---

## 🛠️ Tools & Libraries Used

* **Python**
* **Pandas** – data manipulation
* **NumPy** – numerical operations
* **Seaborn & Matplotlib** – data visualization
* **Scikit-learn** – machine learning (modeling, preprocessing, evaluation)

---

## 📂 Project Workflow

### 1️⃣ Data Loading & Exploration

* Loaded the dataset using **Pandas**
* Performed basic data exploration (shape, info, null values)
* Visualized relationships using **Seaborn**

---

### 2️⃣ Feature & Target Separation

* Separated **independent features (X)** and **target variable (y)**

```python
X = data.drop(columns=["target"])
y = data["target"]
```

---

### 3️⃣ Feature Engineering

#### 🔹 One-Hot Encoding

* Applied **One-Hot Encoding** for categorical variables (e.g., region)
* Used `drop_first=True` to avoid the **dummy variable trap**

```python
pd.get_dummies(X, columns=["region"], drop_first=True)
```

#### 🔹 Binary Encoding

* Converted binary categorical features into numerical format (0/1)

#### 🔹 Interaction Features

* Created interaction features to capture combined effects between variables

```python
X["age_smoker"] = X["age"] * X["smoker"]
```

---

### 4️⃣ Feature Scaling

* Applied **Normalization** and **Standardization** for numeric features
* Especially important where values were on different scales (e.g., salary in lakhs)

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

### 5️⃣ Feature Selection

* Selected important features such as:

  * Score
  * Study Hours
* Reduced noise and improved model performance

---

### 6️⃣ Train-Test Split

* Split data into training and testing sets

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### 7️⃣ Model Training

* Trained the model using **Linear Regression**

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

---

### 8️⃣ Prediction

* Predicted target values using the trained model

```python
y_pred = model.predict(X_test)
```

---

### 9️⃣ Model Evaluation

#### 📊 R² Score

* Measured how well the model explains variance in the target

```python
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
```

#### 📊 Adjusted R²

* Used to account for the number of features in the model

```python
adjusted_r2 = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
```

---

### 🔟 Underfitting & Overfitting Analysis

* Compared **training vs testing performance**
* Evaluated whether the model was:

  * Underfitting (too simple)
  * Overfitting (too complex)

---

## ✅ Key Learnings

* Importance of **feature engineering** in Linear Regression
* Handling categorical variables correctly
* Role of **scaling** in model stability
* Understanding **R² vs Adjusted R²**
* Detecting **underfitting and overfitting**

---

## 🚀 Conclusion

This project strengthened my understanding of **machine learning fundamentals** and how Linear Regression works on real-world data. It highlights the complete ML pipeline—from raw data to model evaluation—using best practices.

---

## 📌 Future Improvements

* Try **Regularization techniques** (Ridge, Lasso)
* Perform **cross-validation**
* Experiment with additional interaction features

---

⭐ If you find this project helpful, feel free to star the repository!.
