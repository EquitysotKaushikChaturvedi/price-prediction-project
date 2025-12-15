import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Smartphone Price Prediction Analysis\n",
    "\n",
    "This notebook covers the exploratory data analysis (EDA), data cleaning, model training, and evaluation for the Smartphone Price Prediction project.\n",
    "\n",
    "## 1. Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
    "\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Load Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('../data/smartphones.csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Data Cleaning & Preprocessing\n",
    "- Fill missing values (Median for Numerical, Mode for Categorical)\n",
    "- Remove outliers using IQR on 'price_inr'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Handling Missing Values\n",
    "num_cols = ['ram_gb', 'storage_gb', 'battery_mah', 'rear_camera_main_mp', 'display_inches']\n",
    "for col in num_cols:\n",
    "    if col in df.columns:\n",
    "        df[col] = df[col].fillna(df[col].median())\n",
    "\n",
    "cat_cols = ['smartphone_brand', 'processor_brand']\n",
    "for col in cat_cols:\n",
    "    if col in df.columns:\n",
    "        df[col] = df[col].fillna(df[col].mode()[0])\n",
    "\n",
    "# Outlier Removal (Price)\n",
    "Q1 = df['price_inr'].quantile(0.25)\n",
    "Q3 = df['price_inr'].quantile(0.75)\n",
    "IQR = Q3 - Q1\n",
    "df_clean = df[(df['price_inr'] >= (Q1 - 1.5 * IQR)) & (df['price_inr'] <= (Q3 + 1.5 * IQR))].copy()\n",
    "\n",
    "print(f\"Original shape: {df.shape}, Cleaned shape: {df_clean.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Visualizations\n",
    "### Price Distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10, 6))\n",
    "sns.histplot(df_clean['price_inr'], bins=30, kde=True)\n",
    "plt.title('Price Distribution')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Correlation Heatmap"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(12, 8))\n",
    "sns.heatmap(df_clean[num_cols + ['price_inr']].corr(), annot=True, cmap='coolwarm')\n",
    "plt.title('Correlation Heatmap')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Model Training\n",
    "We will train Linear Regression and Random Forest Regressor."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df_clean[['smartphone_brand', 'ram_gb', 'storage_gb', 'battery_mah', 'rear_camera_main_mp', 'display_inches', 'processor_brand']]\n",
    "y = df_clean['price_inr']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "numeric_features = ['ram_gb', 'storage_gb', 'battery_mah', 'rear_camera_main_mp', 'display_inches']\n",
    "categorical_features = ['smartphone_brand', 'processor_brand']\n",
    "\n",
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('num', StandardScaler(), numeric_features),\n",
    "        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)\n",
    "    ])\n",
    "\n",
    "models = {\n",
    "    \"Linear Regression\": LinearRegression(),\n",
    "    \"Random Forest\": RandomForestRegressor(n_estimators=100, random_state=42)\n",
    "}\n",
    "\n",
    "results = {}\n",
    "\n",
    "for name, model in models.items():\n",
    "    clf = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])\n",
    "    clf.fit(X_train, y_train)\n",
    "    y_pred = clf.predict(X_test)\n",
    "    \n",
    "    mae = mean_absolute_error(y_test, y_pred)\n",
    "    rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n",
    "    r2 = r2_score(y_test, y_pred)\n",
    "    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'y_pred': y_pred}\n",
    "    \n",
    "    print(f\"{name}:\")\n",
    "    print(f\"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Evaluation Plots\n",
    "### Actual vs Predicted (Random Forest)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "rf_preds = results['Random Forest']['y_pred']\n",
    "plt.figure(figsize=(8, 6))\n",
    "plt.scatter(y_test, rf_preds, alpha=0.5)\n",
    "plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)\n",
    "plt.xlabel('Actual Price')\n",
    "plt.ylabel('Predicted Price')\n",
    "plt.title('Actual vs Predicted (Rankdom Forest)')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Feature Importance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Extract feature importance from RF pipeline\n",
    "rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', models['Random Forest'])])\n",
    "rf_pipeline.fit(X_train, y_train)\n",
    "\n",
    "rf_model = rf_pipeline.named_steps['regressor']\n",
    "preprocessor_step = rf_pipeline.named_steps['preprocessor']\n",
    "\n",
    "cat_names = preprocessor_step.named_transformers_['cat'].get_feature_names_out(categorical_features)\n",
    "feature_names = np.r_[numeric_features, cat_names]\n",
    "\n",
    "importances = rf_model.feature_importances_\n",
    "indices = np.argsort(importances)[::-1]\n",
    "\n",
    "top_n = 20\n",
    "plt.figure(figsize=(10, 8))\n",
    "plt.title(\"Feature Importances\")\n",
    "plt.barh(range(top_n), importances[indices[:top_n]], align=\"center\")\n",
    "plt.yticks(range(top_n), feature_names[indices[:top_n]])\n",
    "plt.gca().invert_yaxis()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open(r'd:\price-prediction-project\notebooks\analysis.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=1)
