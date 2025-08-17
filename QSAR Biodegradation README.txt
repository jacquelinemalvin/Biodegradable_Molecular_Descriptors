# QSAR Biodegradation - Classification with Feature Plots

## Overview
This notebook demonstrates a simple machine learning workflow for predicting whether a chemical compound is **readily biodegradable** using the QSAR Biodegradation dataset.  
Two classification algorithms are used:
- **Logistic Regression**
- **Random Forest Classifier**

The notebook also includes **feature importance** and **coefficient** plots to visualize how each feature contributes to predictions.

---

## Steps Performed

1. **Load Dataset**
   - Reads the `qsar_biodeg.csv` file containing molecular descriptors and the target biodegradability class.

2. **Identify Features and Target**
   - All columns except the target column are used as features (`X`).
   - The target column (`y`) is converted to binary:
     - `1` = Readily biodegradable (RB)
     - `0` = Not readily biodegradable (NRB)

3. **Split Data**
   - Uses `train_test_split` to divide the dataset into training (80%) and testing (20%) sets.
   - `stratify=y` ensures class proportions are maintained.
   - `random_state=42` makes the split reproducible.

4. **Scale Features**
   - Standardizes feature values using `StandardScaler` for models sensitive to feature magnitude (e.g., Logistic Regression).

5. **Train Models**
   - **Logistic Regression**:
     - Fits on scaled features.
     - Outputs accuracy and classification report.
   - **Random Forest Classifier**:
     - Fits on unscaled features (tree-based models don’t require scaling).
     - Outputs accuracy and classification report.

6. **Visualize Model Insights**
   - **Random Forest Feature Importance Plot**:
     - Shows which features were most influential in the Random Forest model.
   - **Logistic Regression Coefficient Plot**:
     - Displays how each feature affects the probability of the positive class.

---

## How to Run
1. Place `qsar_biodeg.csv` in the same folder as the notebook or update `csv_path` to point to the dataset location.
2. Install required Python packages:
   ```bash
   pip install pandas matplotlib scikit-learn
