<div align="justify">

**"One cannot create a mosaic without the hard small marble bits known as 'facts' or 'data'; what matters, however, is not so much the individual bits as the sequential patterns into which you organize them, then break them up and reorganize them."**
</div>
<div align="right">
  — Timothy Robinson
</div>
<br>

<p align="center">
  <img src="Exploratory Data Analysis/Images/My Handbook title.png" alt="Data Demystified Cover" width="600"/>
</p>

<div align="justify">

In the realm of machine learning, **data is not just fuel — it is the foundation of insight**. But like scattered mosaic tiles, raw data is rarely useful in its original form. It’s fragmented, noisy, and often misleading.

This handbook is your blueprint for **turning those fragments into form** — for finding structure in disorder, meaning in mess, and patterns in chaos. Whether you’re decoding a messy CSV or refining structured logs, the goal remains the same: to **reorganize raw facts into intelligent frameworks** that models can trust and learn from.

This is not a mere cookbook of syntax. It is a guided journey through:
- Exploring the unknowns of a dataset  
- Cleaning and reconciling inconsistencies  
- Engineering features that capture signal, not noise  
- Selecting what truly matters  
- Shaping data for compatibility, consistency, and clarity  
</div>
<p align="center">
  <img src="Exploratory Data Analysis/Images/Workflow.png" alt="Data Demystified Cover" width="600"/>
</p>

### 🔍 Section 1: Exploratory Data Analysis (EDA)
<div align="justify">
  
Exploratory Data Analysis (EDA) is a pivotal step in the data science pipeline. It involves systematically examining the dataset to uncover patterns, spot anomalies, test assumptions, and generate hypotheses — **using both statistical summaries and visualizations.**
<br>

EDA is the **first meaningful conversation with your data.** It reveals hidden structures and contextual insights that guide every step that follows — from cleaning and transformation to feature selection and model building. 
Before you trust any algorithm, you must understand the data it learns from. 
This foundational process shapes the integrity and success of your entire project.
<br>

Coined by John Tukey, EDA is the art of "letting the data speak for itself."
</div>

#### Core Objectives of EDA:

- Understand the structure and types of data (categorical, numerical, datetime, etc.)
- Detect missing values, outliers, and inconsistencies
- Visualize feature distributions and assess skewness
- Explore correlations and potential multicollinearity
- Validate assumptions for modeling (e.g., normality, linearity)

#### Key Steps in EDA:

| **Step**                  | **Purpose**                                                     |
|---------------------------|------------------------------------------------------------------|
| Dataset Overview          | Understand basic structure, dimensions, and data types          |
| Missing Value Detection   | Identify and address incomplete or null data                    |
| Data Quality Inspection   | Find duplicates, inconsistencies, or formatting issues          |
| Outlier Detection         | Spot extreme values that can distort analysis                   |
| Univariate Analysis       | Explore individual variable distributions                       |
| Bivariate / Multivariate  | Examine relationships between two or more variables             |
| Correlation Analysis      | Uncover linear relationships between numeric variables          |

#### Importing Libraries

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
```

#### Loading the Dataset
```python
df = pd.read_csv('your_dataset.csv')
df.head()
```

#### 1.1 Dataset Overview:

```python
Shape of data         : df.shape
Data types            : df.dtypes
Column names          : df.columns
Sample records        : df.head(), df.tail()
Descriptive statistics: df.describe()
Info summary          : df.info()
```

#### 1.2 Missing Value Detection:
```python
df.isnull().sum()
df.isnull().mean() * 100    # Percentage missing
msno.matrix(df)             # visualize missing values
```

#### 1.3 Check for Duplicates:
```python
df.duplicated().sum()
```

#### 1.4 Outlier Detection:
```python
# IQR Method
# Rule: Outliers lie outside [Q1 - 1.5 × IQR, Q3 + 1.5 × IQR]

Q1 = df['feature'].quantile(0.25)
Q3 = df['feature'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['feature'] < Q1 - 1.5 * IQR) | (df['feature'] > Q3 + 1.5 * IQR)]

# Z-Score Method | Standardizes data | Best for normally distributed data.
# Rule: Z-score > 3 or < -3 is considered an outlier.

from scipy.stats import zscore
df['z_score'] = zscore(df['feature'])
outliers = df[(df['z_score'] > 3) | (df['z_score'] < -3)]

# Boxplot Visualization (Outliers appear as points beyond whiskers.)
import seaborn as sns
sns.boxplot(y=df['feature'])
```

#### 📊 1.5 Visualizations Used in EDA
####  1. <u>Univariate Analysis (Single Variable)</u>

**For Numerical Variables:**
- Histogram   → `sns.histplot(df['col'])`
- Boxplot&nbsp;&nbsp;&nbsp;&nbsp;  → `sns.boxplot(y=df['col'])`
- Violin Plot → `sns.violinplot(y=df['col'])`
- KDE Plot   &nbsp;&nbsp;→ `sns.kdeplot(df['col'])`

**For Categorical Variables:**
- Count Plot → `sns.countplot(x=df['col'])`
  
#### 2. <u>Bivariate Analysis (Two Variables)</u>

**Numerical vs Numerical:**
- Scatter Plot → `sns.scatterplot(x='col1', y='col2', data=df)`
- Hexbin Plot → `df.plot.hexbin(...)`
- Line Plot &nbsp;&nbsp;&nbsp;&nbsp; → `sns.lineplot(...)`

**Categorical vs Numerical:**
- Boxplot &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ `sns.boxplot(x='cat_col', y='num_col', data=df)`
- Violin Plot&nbsp; → `sns.violinplot(...)`
- Swarm Plot → `sns.swarmplot(...)`

**Categorical vs Categorical:**
- Grouped Bar Plot &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ `sns.countplot(x='cat1', hue='cat2', data=df)`
- Heatmap of Crosstab → `sns.heatmap(pd.crosstab(...))`

#### 3. Multivariate Analysis (3+ Variables)

- **Pairplot**&nbsp;&nbsp;&nbsp; → `sns.pairplot(df[['num1','num2','target']], hue='target')`
- **Heatmap**&nbsp; → `sns.heatmap(df.corr())`

>📌 For practical examples and hands-on implementation of all Seaborn plots mentioned in this guide, refer to my [Seaborn Visualization Notebook](https://github.com/Smarth2005/UCS420-Cognitive-Computing-Lab-Assignments/blob/main/Python%20Basic%20Libraries%3A%20Hands-On%20Implementation/L4.Seaborn.ipynb)

#### 1.6 Feature Distribution & Skewness:
Understanding the distribution shape (normal, skewed) helps decide transformations (e.g., log, square root).
Detect Skewness:
```python
df['feature'].skew()                   # +ve skew → right tail, -ve skew → left tail
sns.histplot(df['feature'], kde=True)
```

#### 1.7 Correlation Analysis: Understand relationships between 'numerical' variables.
```python
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
```

Use `.corr()` with:
- `pearson` (default): linear
- `kendall`, `spearman`: rank-based (non-linear)
📌 Remove highly correlated features (correlation > 0.9) to avoid multicollinearity in models.

####  1.8 Multicollinearity Check (Optional, for Modeling Readiness)
Check variance inflation factors (VIF) if you're preparing for linear models.
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = df[['feature1', 'feature2', 'feature3']]
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
```
📌 A VIF > 5 or 10 often indicates multicollinearity.

---

### Section 2: Data Cleaning
<div align="justify">
Data Cleaning is the process of fixing or removing incorrect, corrupted, misformatted, duplicate, or incomplete data within a dataset.
High-quality data is essential for building robust models, and poor data hygiene is one of the most common causes of poor model performance.

</div>

#### Core Tasks in Data Cleaning:
| **Task**                   | **Purpose**                                                |
| -------------------------- | ---------------------------------------------------------- |
| Handling Missing Values    | Decide whether to drop, fill, or flag missing data         |
| Removing Duplicates        | Prevent overfitting and bias from repeated records         |
| Fixing Data Types          | Ensure correct formats (e.g., dates as datetime, numerics) |
| Formatting Inconsistencies | Standardize units, spellings, or category names            |
| Outlier Handling           | Remove or treat extreme values that affect distribution    |
| Categorical Normalization  | Convert inconsistent category names or encode properly     |
| Date/Time Conversion       | Convert to datetime objects for time series analysis       |

#### 2.1 Handling Missing Values 
**Strategies:**
- Drop Rows with many missing entries:
  ```python
  df.dropna(subset=['important_col'], inplace=True)
  ```

- Drop Columns with too many missing values:
  ```python
  df.drop(columns=['unusable_column'], inplace=True)
  ```

- Impute with mean/median/mode:
  ```python
  df['col'].fillna(df['col'].mean(), inplace=True)       # for numerical features
  df['col'].fillna(df['col'].mode()[0], inplace=True)    # for categorical features
  ```
Use domain knowledge + EDA to decide what’s appropriate.

#### 2.2 Removing Duplicates
```python
df.duplicated().sum()           # total duplicates
df.drop_duplicates(inplace=True)
```

#### 2.3 Fixing Incorrect Datatypes
```python
df['date'] = pd.to_datetime(df['date'])     # convert string to datetime
df['num'] = pd.to_numeric(df['num'])        # convert to float or int
```
>  Often happens when numeric data is loaded as string due to symbols like % or $.

#### 2.4 Fixing Inconsistent Categorical Values
```python
df['city'].unique()
df['city'] = df['city'].str.lower().str.strip()
```
> Example: `'New York ', ' new york', 'NEW YORK'` → all become `new york`.

#### 2.5 Handling Outliers
If not handled in EDA, now is the time to:
- Clip them
- Remove them
- Transform them
```python
df['feature'] = df['feature'].clip(lower_bound, upper_bound)
```

---

### Section 3: Feature Engineering
<div align="justify">
Feature engineering involves creating, transforming, or selecting input features that can improve model performance. One of the first and most common steps in this phase is encoding — the process of converting categorical variables into a numerical format that machine learning models can interpret.

</div>

#### 3.1 Encoding Categorical Variables
#### 📌 Why Encoding?
Most machine learning algorithms can’t handle categorical variables directly. Encoding ensures they are translated into meaningful numeric representations without distorting relationships or introducing bias.

#### Types of Encoding:
| **Encoding Method**  | **Use When**                                                      | **Suitable For**              |
| -------------------- | ----------------------------------------------------------------- | ----------------------------- |
| **Label Encoding**   | Categories have a natural order                                   | Tree-based models             |
| **One-Hot Encoding** | Categories are nominal (no inherent order)                        | Linear models, most models    |
| **Ordinal Encoding** | Categories are ordinal (low < medium < high)                      | Tree-based or linear models   |
| **Target Encoding**  | You want to capture relationship with the target (risk: leakage!) | Carefully in supervised tasks |
| **Binary Encoding**  | High cardinality categorical features (many unique categories)    | High-dimensional data         |

#### (i) Label Encoding:
Assigns a unique integer to each category. Good for ordinal data or tree-based models that are insensitive to magnitude.
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['encoded_col'] = le.fit_transform(df['col'])
```

#### (ii) One-Hot Encoding:
It converts each category into a new binary (0/1) column, representing presence or absence of that category.
Best for nominal (unordered) categories.
```python
df = pd.get_dummies(df, columns=['col'], drop_first=True)
# `drop_first`=True avoids multicollinearity by dropping the first column.
```
<div align="center">
  
  **OR**
</div>

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Sample data
df = pd.DataFrame({'City': ['Delhi', 'Mumbai', 'Delhi', 'Bangalore']})

# Initialize encoder
encoder = OneHotEncoder(sparse=False)  # sparse=False to return an array, not sparse matrix

# Fit and transform
encoded = encoder.fit_transform(df[['City']])

# Create encoded DataFrame with column names
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['City']))

# Combine with original
final_df = pd.concat([df, encoded_df], axis=1)
```

#### (iii) Ordinal Encoding:
Map categories to meaningful integer values — best used when there's natural ranking.
```python
from sklearn.preprocessing import OrdinalEncoder

order = [['low', 'medium', 'high']]
enc = OrdinalEncoder(categories=order)
df['encoded'] = enc.fit_transform(df[['col']])
```

#### (iv) Target Encoding:
Replace categories with the mean of the target for each category. Use carefully to avoid data leakage.

```python
mean_encoded = df.groupby('col')['target'].mean()
df['col_encoded'] = df['col'].map(mean_encoded)
```
⚠️ Use cross-validation or out-of-fold means to avoid leakage in real tasks.

#### (v) Binary Encoding: (Optional - for High Cardinality)
Efficient when you have many unique categories.

Install first: `pip install category_encoders`

```python
import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['high_card_col'])
df_encoded = encoder.fit_transform(df)
```
#### Best Practices for Encoding:
- Use One-Hot Encoding for models like Logistic Regression, SVM, KNN.
- Use Label or Ordinal Encoding for Decision Trees, Random Forests, XGBoost.
- Avoid Target Encoding unless you know what you’re doing (risk of overfitting).
- Always check cardinality before encoding — high cardinality → binary or frequency encoding.

#### 3.2 Feature Transformation
<div align="justify"> 
Feature transformation refers to the process of modifying existing features to improve the model’s ability to learn patterns from data. It helps address skewed distributions, normalize scales, and create new insights from raw data. 
</div>

#### Common Feature Transformation Techniques:
| **Transformation**        | **When to Use**                                                      | **Purpose**                               |
| ------------------------- | -------------------------------------------------------------------- | ----------------------------------------- |
| **Log Transformation**    | Data is right-skewed (e.g., income, prices)                          | Normalize distribution, reduce variance   |
| **Min-Max Scaling**       | Features vary widely in scale                                        | Normalize to \[0,1] range                 |
| **Standard Scaling**      | Gaussian-like distributions                                          | Center around mean 0, std dev 1           |
| **Robust Scaling**        | Presence of outliers                                                 | Scale using median & IQR (less sensitive) |
| **Box-Cox / Yeo-Johnson** | Skewed continuous data (Box-Cox for positive only, YJ supports zero) | Normalize distribution                    |
| **Binning**               | Convert continuous into categorical                                  | Handle non-linearity, reduce noise        |

#### (i) Log Transformation
```python
import numpy as np

df['log_feature'] = np.log1p(df['feature'])   # log(1 + x) avoids log(0) error
```
#### (ii) Min-Max Scaling (Normalization)
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['scaled_col']] = scaler.fit_transform(df[['col']])
```

#### (iii) Standard Scaling (Z-score Normalization)
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['standardized']] = scaler.fit_transform(df[['col']])
```

#### (iv) Robust Scaling (Median & IQR)
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
df[['robust_scaled']] = scaler.fit_transform(df[['col']])
```

#### (v) Binning / Discretization
Split continuous variables into bins:

```python
df['binned'] = pd.cut(df['col'], bins=4, labels=["low", "medium", "high", "very high"])
```
Can also be used with pd.qcut() for quantile-based binning.

#### (vi) Power Transformations (Box-Cox, Yeo-Johnson)
```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson')  # 'box-cox' works only for positive values
df[['transformed']] = pt.fit_transform(df[['col']])
```

#### When to Apply Feature Transformation?
- Before applying distance-based models like KNN, SVM, or neural networks
- After encoding categorical variables if required
- On numerical variables with skewed distributions or high variance
- Before modeling, as part of your preprocessing pipeline

#### 3.3 Feature Selection
<div align="justify">
Feature Selection is the process of identifying and selecting the most relevant features from your dataset that contribute the most to the prediction variable.
It reduces dimensionality, eliminates noise, improves model performance, and enhances interpretability.

</div>

#### Why Feature Selection Matters:
- Reduces overfitting
- Improves model accuracy
- Decreases training time
- Removes irrelevant or redundant features

#### Approaches to Feature Selection:
| **Category** | **Method**                      | **Description**                                |
| ------------ | ------------------------------- | ---------------------------------------------- |
| **Filter**   | Correlation, Chi-square, ANOVA  | Select features based on statistical scores    |
| **Wrapper**  | Forward/Backward Selection, RFE | Evaluate subsets using a predictive model      |
| **Embedded** | Lasso, Tree-based importance    | Feature selection occurs during model training |

#### (i) Filter Methods (Model-Independent)
#### ➤ Correlation Matrix (For numerical features)
```python
import seaborn as sns
import matplotlib.pyplot as plt

corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
```
📌 Drop highly correlated features (e.g., r > 0.85).

#### ➤ Chi-Square Test (For categorical features vs. categorical target)
```python
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

# Encode categories
X = df[['feature1', 'feature2']]
y = LabelEncoder().fit_transform(df['target'])

chi_scores = chi2(X, y)
```

#### ➤ ANOVA F-test (For numerical features vs. categorical target)
```python
from sklearn.feature_selection import f_classif

f_values, p_values = f_classif(X, y)
```

#### (ii) Wrapper Methods (Model-Based Search)
#### ➤ Recursive Feature Elimination (RFE)
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=5)
fit = rfe.fit(X, y)
selected_features = X.columns[fit.support_]
```

#### ➤ Forward / Backward Feature Selection (with mlxtend)
`pip install mlxtend`
<br>
```python
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression

sfs = SFS(LogisticRegression(), 
          k_features=5, 
          forward=True, 
          floating=False,
          scoring='accuracy',
          cv=5)

sfs = sfs.fit(X, y)
```

#### (iii) Embedded Methods
#### ➤ L1 Regularization (Lasso Regression)
```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
model.fit(X, y)
selected_features = X.columns[model.coef_ != 0]
```

#### ➤ Feature Importance from Tree-Based Models
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X, y)

importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).plot(kind='bar')
```

#### Best Practices for Feature Selection:
- Visualize correlations to identify redundancy.
- Use filter methods for initial reduction.
- Use wrapper/embedded methods to refine selection based on model performance.
- Perform cross-validation after feature selection to validate impact.

### 🎉 Wrapping Up
<div align="justify">
With exploratory analysis complete, missing values handled, features encoded and scaled, and relevant variables selected, we now conclude the data preparation phase.  
At this point, your dataset is clean, transformed, and model-ready — ready to be fed into any machine learning algorithm with confidence.
</div>

---

### Closing Remarks
<div align="justify">
  
Data preparation is not just the first phase of a machine learning project — it’s the **foundation**.  
The accuracy, efficiency, and interpretability of your models hinge on the quality of your preprocessing. Whether you're building a prototype or preparing for deployment, remember:
<br>

**“The model learns from your data. So first, you must learn from your data.”**

This handbook is meant to be a **practical reference** and a **learning guide**. Revisit it as needed during real-world projects — and may your data always be clean, your features meaningful, and your models insightful.
</div>

###  Author's Section

This handbook reflects my ongoing journey in understanding, documenting, and sharing the foundational steps of the data science lifecycle.
It is both a personal knowledge archive and a contribution to fellow learners navigating the ML pipeline.

**Smarth Kaushal** <br>
Student, Pre-Final Year, BE-CSE <br>
Thapar Institute of Engineering and Technology, Patiala <br>
Computer Science and Engineering Department

Email: skaushal1_be23@thapar.edu <br>
Connect on [LinkedIn](http://www.linkedin.com/in/smarth-kaushal-02a1092b2) <br>
Collaborate via [GitHub](https://github.com/Smarth2005)



