# Categorical Data Encoding
## `pd.get_dummies()` vs `OneHotEncoder()`

This project is part of my **Data Science learning journey**, where I explore different preprocessing techniques used in machine learning.

The goal of this notebook is to compare two popular methods for encoding categorical variables:

- `pandas.get_dummies()`
- `sklearn.preprocessing.OneHotEncoder`

Both methods are applied to the **same synthetic Netflix dataset**, allowing for a fair comparison of their workflow, usability, and performance when training a machine learning model.

---

## Project Objective

Categorical variables cannot be used directly by most machine learning algorithms. This notebook investigates two common approaches to converting categorical features into numerical representations and compares their results.

The comparison focuses on:

- Ease of implementation
- Integration with machine learning workflows
- Handling of unseen categories
- Model performance

---

## Dataset

A synthetic Netflix-style dataset containing **1,000 movies**.

### Features

- Genre
- Country
- Language
- Age Rating
- Release Decade
- Subscription Plan
- Device
- Weekend Release
- Duration
- IMDb Score
- Production Budget

### Target

- `target_popular`
  - `1` = Popular
  - `0` = Not Popular

---

## Workflow

### Method 1 – `pd.get_dummies()`

- Split the dataset into training and testing sets
- Encode categorical variables using `pd.get_dummies()`
- Align train and test columns using `reindex()`
- Train a Random Forest classifier
- Evaluate the model

---

### Method 2 – `OneHotEncoder()`

- Split the dataset into training and testing sets
- Create a preprocessing pipeline using `ColumnTransformer`
- Encode categorical variables with `OneHotEncoder(handle_unknown="ignore")`
- Train the same Random Forest classifier
- Evaluate the model

---

## Results

| Metric | `pd.get_dummies()` | `OneHotEncoder()` |
|---------|--------------------|-------------------|
| Accuracy | **0.915** | **0.915** |

Both encoding methods achieved the same classification accuracy on this dataset.

---

## Key Takeaways

### `pd.get_dummies()`

**Advantages**
- Very simple to use
- Ideal for exploratory analysis
- Produces a pandas DataFrame

**Limitations**
- Requires manual column alignment between training and testing data
- Less suitable for production machine learning pipelines

---

### `OneHotEncoder()`

**Advantages**
- Integrates seamlessly with scikit-learn Pipelines
- Automatically applies the same encoding to new data
- Handles unseen categories using `handle_unknown="ignore"`
- Better suited for scalable machine learning workflows

**Limitations**
- Slightly more verbose to implement
- Returns a sparse matrix by default instead of a pandas DataFrame

---

## Conclusion

Although both encoding methods produced identical predictive performance in this experiment, they differ significantly in their workflow.

`pd.get_dummies()` is an excellent choice for quick exploratory analysis and smaller projects due to its simplicity. However, `OneHotEncoder()` offers a more robust and reusable solution for machine learning pipelines by automatically handling feature consistency and unseen categories.

This exercise helped me understand not only how both techniques work, but also **when each approach is most appropriate** in a real-world machine learning project.

---

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- Jupyter Notebook

---

> **Note:** This project was created for educational purposes as part of my data science learning journey ;).