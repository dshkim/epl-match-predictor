# EPL Match Predictor

## Introduction

This project combines a love for soccer with data science!

### Background

Soccer has always been a major part of my life. I grew up playing for an academy team and competed at the highest level; participating in tournaments and competing with the likes of MLS academy teams. From the thrill of watching EPL matches to debating with friends about team performances and player statistics, my enthusiasm for the game runs deep. This project is a reflection of my commitment to understanding soccer not just as a fan, but through the lens of data and analytics.

This project represents the intersection of my passion for soccer and my interest in data science. By scraping detailed EPL match data and applying predictive modeling techniques, I’ve turned my fascination with the sport into a practical exploration of data. The process of transforming raw match data into meaningful predictions has been both challenging and exhilarating, aligning perfectly with my goals of applying analytical skills to real-world scenarios.

### Future Directions

Looking ahead, there are exciting possibilities for this project. One potential avenue is developing a recommendation system for sports betting. By enhancing the current model and incorporating additional data, I aim to create a tool that offers valuable insights and recommendations for betting enthusiasts. This evolution will deepen my engagement with soccer analytics and expand the practical applications of this work.

Thank you for checking out this project. Whether you’re passionate about soccer, interested in data science, or curious about sports predictions, I hope you find this work as fascinating and rewarding as I do!

## Data Collection

### Data Source

The data for this project was sourced from [fbref.com](https://fbref.com/en/comps/9/Premier-League-Stats), a comprehensive website for soccer statistics. This platform provides detailed performance metrics and match data for the English Premier League (EPL).

### Data Range

I collected data spanning from the 2019-2020 season through to the 2023-2024 season. At the time of collection, the 2024-2025 season had just begun, so it was not included in this dataset.

### Data Collection Process

To gather the data, I developed an automated Python web scraping script [`epl-data-web-scraper.py`](https://github.com/dshkim/epl-match-predictor/blob/main/web-scraping/epl-data-web-scraper.py). 

The process involved:

1. **Requesting Web Content**: Using the `requests` library, the script sends HTTP requests to the fbref.com pages containing EPL match data.
2. **Parsing HTML**: The received web content is then converted into HTML format.
3. **Extracting Data**: `BeautifulSoup` is used to parse the HTML and extract relevant match statistics and performance metrics.

### Potential Improvements

For future enhancements, I plan to incorporate batch processing to automate the update of new match data as it becomes available. This would allow for continuous data collection and keep the dataset up-to-date with the latest matches and statistics.

## Data Preprocessing/Cleaning

To prepare the dataset for machine learning model development, several preprocessing and cleaning steps were carried out to ensure data quality and consistency.

Follow this notebook for more detailed steps: [`EDA.ipynb`](https://github.com/dshkim/epl-match-predictor/blob/main/data-cleaning/EDA.ipynb). 

### Data Cleaning

1. **Column Dropping**: Columns that were not necessary for predictions or were directly related to match outcomes were removed. These included:
   - `comp` (Competition)
   - `gf` (Goals For)
   - `ga` (Goals Against)
   - `xg` (Expected Goals)
   - `xga` (Expected Goals Against)
   - `attendance` (Match Attendance)
   - `match report` (Match Report)
   - `notes` (Additional Notes)
   - `season` (Season)

2. **Handling Missing Data**: Rows with missing data were dropped. Only one row had missing values after dropping unnecessary, which was an outlier and indicated that the data collection process was generally successful.

3. **Standardizing Data**: Team names were standardized to maintain consistency across the dataset. For instance, "Manchester United" was changed to "Manchester Utd" to ensure uniformity between team and opponent columns.

4. **Date Conversion**: Date fields were converted to `datetime` objects to facilitate time-based analysis.

5. **Data Type Conversion**: Ensured that all columns followed the correct data types, with numbers converted to integers as needed.

6. **Outcome Encoding**: Match outcomes were encoded into numerical values:
   - `'W'` (Win) became `1`
   - `'L'` (Loss) became `-1`
   - `'D'` (Draw) became `0`

These preprocessing steps were crucial for creating a dataset that is clean, consistent, and suitable for training a machine learning model.

## Feature Selection/Engineering

To develop a robust machine learning model, careful feature selection and engineering were conducted to ensure the most relevant data was used. 

Follow this notebook for more detailed steps: [`EDA.ipynb`](https://github.com/dshkim/epl-match-predictor/blob/main/data-cleaning/EDA.ipynb). 

### Selected Features

The following columns were chosen for training the model:

- `time` (Time of the Match)
- `round` (Match Round)
- `day` (Day of the Week)
- `venue` (Match Venue)
- `opponent` (Opponent Team)
- `poss` (Possession Percentage)
- `formation` (Team Formation)
- `opp formation` (Opponent Formation)
- `referee` (Match Referee)
- `sh` (Shots)
- `sot` (Shots on Target)
- `dist` (Shot Distance)
- `fk` (Free Kicks)
- `pk` (Penalties)
- `pkatt` (Penalty Attempts)
- `team` (Team Name)

### Feature Selection Process

Feature selection was driven primarily by domain expertise and an understanding of which features are likely to impact match outcomes. This approach ensures that the model is trained with relevant data that can influence the results of the matches.

### Feature Engineering

- **One-Hot Encoding (OHE)**: Categorical variables, such as `venue`, `opponent`, `formation`, `opp formation`, and `referee`, were converted into numerical format using one-hot encoding. This transformation allows the model to process categorical data effectively.

These steps in feature selection and engineering helped to focus the model on the most impactful variables and prepare the data for effective machine learning analysis.

## Model Selection

Choosing the right model is crucial for achieving accurate predictions. For this project, the focus was on selecting models that could handle the non-linear relationships present in the data and deliver robust performance.

### Model Choices

#### Random Forest

**Why Random Forest?**
- **Nonparametric Nature**: Random Forest is a nonparametric model that does not make assumptions about the underlying distribution of the data. This flexibility is particularly useful for capturing complex, non-linear relationships between features and match outcomes.
- **Robustness**: The model is robust to overfitting, especially when dealing with high-dimensional data. By averaging predictions from multiple decision trees, Random Forest reduces variance and improves generalization.
- **Feature Importance**: Random Forest provides insights into feature importance, helping to understand which variables have the most influence on match outcomes.

#### XGBoost

**Why XGBoost?**
- **Handling Non-Linearity**: XGBoost is also a nonparametric model that excels in capturing complex, non-linear relationships. It uses gradient boosting to enhance performance and accuracy, making it suitable for datasets with intricate patterns.
- **Efficiency and Performance**: XGBoost is known for its efficiency and scalability, offering high performance even on large datasets. It incorporates techniques like regularization to prevent overfitting and optimize model accuracy.
- **Flexibility**: The model allows for fine-tuning of hyperparameters, which helps in achieving the best possible performance for the specific characteristics of the data.

### Why Not Logistic Regression?

Logistic Regression was not chosen due to its linearity assumption, which is not well-suited for capturing the non-linear relationships between the features and match outcomes in this dataset. The non-linear models selected—Random Forest and XGBoost—offer greater flexibility and robustness for this type of prediction task.

By leveraging Random Forest and XGBoost, the model is better equipped to handle the complexities of the EPL match data and provide more accurate predictions of match outcomes.

## Model Training

Training the model involved addressing the challenges posed by the time series nature of the data. Ensuring that the chronological order of matches was preserved was crucial for building a reliable predictive model.

Follow this notebook for more detailed steps on the Random Forest model: [`random-forest.ipynb`](https://github.com/dshkim/epl-match-predictor/blob/main/models/random-forest.ipynb). 

Follow this notebook for more detailed steps on the XGBoost model: [`xgboost.ipynb`](https://github.com/dshkim/epl-match-predictor/blob/main/models/xgboost.ipynb). 

### Time Series Considerations

Given that matches are organized in chronological order, it was essential to maintain this order when splitting the data for training and testing. Specifically:
- **Chronological Splitting**: To avoid using future matches to predict past outcomes, the data was split based on time. Matches from a specific cutoff date were used for training, while matches from after this date were used for testing.

### Data Split

- **Train-Test Split**: The dataset was split with 80% used for training and 20% for testing. The cutoff date was determined such that 80% of the data (up to this date) was allocated to the training set, and the remaining 20% (after this date) was used for testing.

### Model Parameters

#### Random Forest

The Random Forest model was configured with the following parameters:
- `n_estimators=100` (Number of trees in the forest)
- `min_samples_split=10` (Minimum number of samples required to split an internal node)
- `random_state=42` (Seed for random number generation to ensure reproducibility)

#### XGBoost

The XGBoost model was created with these parameters:
- `objective='multi:softmax'` (Objective function for multi-class classification)
- `num_class=3` (Number of classes: win, loss, draw)
- `eval_metric='mlogloss'` (Evaluation metric for multi-class classification)

These parameters were later optimized during hyperparameter tuning

## Model Evaluation

The task at hand involves a multi-class classification problem where the goal is to predict the outcome of EPL matches as either a win, loss, or draw. To evaluate the performance of the models, several metrics were used to assess their effectiveness in classifying these outcomes.

Follow this notebook for more detailed steps on the Random Forest model: [`random-forest.ipynb`](https://github.com/dshkim/epl-match-predictor/blob/main/models/random-forest.ipynb). 

Follow this notebook for more detailed steps on the XGBoost model: [`xgboost.ipynb`](https://github.com/dshkim/epl-match-predictor/blob/main/models/xgboost.ipynb). 

### Evaluation Metrics

- **Accuracy**: A primary metric that indicates the proportion of correct predictions. However, it may not be sufficient alone, especially in cases of class imbalance.
- **Precision**: Measures the proportion of true positive predictions among all positive predictions made.
- **Recall**: Indicates the proportion of true positive predictions among all actual positives.
- **F1-Score**: The harmonic mean of precision and recall, providing a balanced measure of a model's performance.

### Model Performance

#### Random Forest

- **Accuracy**: 0.60
- **Detailed Metrics**:
  - **Win (1)**: Precision = 0.60, Recall = 0.80, F1-Score = 0.68
  - **Draw (0)**: Precision = 0.22, Recall = 0.02, F1-Score = 0.04
  - **Loss (-1)**: Precision = 0.62, Recall = 0.71, F1-Score = 0.66
  - **Macro Average**: Precision = 0.48, Recall = 0.51, F1-Score = 0.46
  - **Weighted Average**: Precision = 0.53, Recall = 0.60, F1-Score = 0.54

#### XGBoost

- **Accuracy**: 0.57
- **Detailed Metrics**:
  - **Win (2)**: Precision = 0.64, Recall = 0.63, F1-Score = 0.64
  - **Draw (0)**: Precision = 0.30, Recall = 0.17, F1-Score = 0.22
  - **Loss (1)**: Precision = 0.59, Recall = 0.73, F1-Score = 0.65
  - **Macro Average**: Precision = 0.51, Recall = 0.51, F1-Score = 0.50
  - **Weighted Average**: Precision = 0.55, Recall = 0.57, F1-Score = 0.55

### Comparison and Insights

- **Accuracy**: The Random Forest model slightly outperforms XGBoost with an accuracy of 0.60 compared to 0.57. However, accuracy alone does not provide a complete picture, especially with class imbalance.

- **Precision and Recall**:
  - **Random Forest** shows strong performance in predicting losses and wins but struggles with draws, which are predicted with lower precision and recall.
  - **XGBoost** has better precision and recall for predicting wins and losses compared to draws, similar to Random Forest. However, XGBoost performs slightly better in predicting draws compared to Random Forest.

- **F1-Score**:
  - **Random Forest** has a higher F1-Score for predicting losses and wins, but the F1-Score for draws is very low, indicating that the model struggles with this class.
  - **XGBoost** has a more balanced F1-Score across the classes, especially improving slightly in the prediction of draws compared to Random Forest.

Overall, both models demonstrate strengths and weaknesses. Random Forest performs better in classifying wins and losses, while XGBoost offers more balanced performance across different outcomes. Further tuning and potentially combining these models could improve overall performance and handle class imbalances more effectively.

## Model Improvement

To enhance the performance of the prediction models, several strategies were employed, including hyperparameter tuning and the introduction of rolling averages for feature engineering.

### Hyperparameter Tuning

Hyperparameter tuning was conducted to optimize the models by identifying the most effective parameter settings. Grid Search was used for this purpose, allowing for an exhaustive search over specified parameter values.

- **XGBoost**: The Grid Search for XGBoost explored various combinations of `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree`. The optimal parameters found were `colsample_bytree=0.9`, `learning_rate=0.1`, `max_depth=3`, `n_estimators=100`, and `subsample=0.8`. These settings improved the model's accuracy to 61.6%, demonstrating enhanced performance in predicting match outcomes.

- **Random Forest**: For Random Forest, Grid Search was applied to parameters such as `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`. The best parameters identified were `max_depth=None`, `min_samples_leaf=4`, `min_samples_split=2`, and `n_estimators=100`. These settings resulted in an accuracy of 60.1%, showing improved performance, particularly in predicting match results.

Grid Search was chosen for its effectiveness in exhaustively evaluating parameter combinations, providing a structured approach to optimizing model performance.

### Incorporating Rolling Averages

To further refine the Random Forest model, rolling averages were introduced for numerical features. The optimal window for rolling averages was determined to be the last 2 matches. This technique smooths out short-term fluctuations and highlights longer-term trends.

**Benefits of Rolling Averages**:
- **Trend Identification**: Rolling averages help capture temporal patterns by smoothing out noise and emphasizing significant trends, which can be crucial for predicting outcomes in time series data.
- **Enhanced Feature Representation**: By incorporating rolling averages of the last 2 matches, the model can better understand performance trends and dynamics over time, leading to potentially more accurate predictions.

Overall, these improvements aimed to enhance model accuracy and robustness by optimizing hyperparameters and incorporating additional features that reflect underlying data patterns.

## Conclusion

This project involved developing a machine learning model to predict the outcomes of English Premier League (EPL) matches utilizing past seasons data. Through a series of steps, including data collection, preprocessing, feature engineering, model selection, training, and improvement, I was able to create a robust predictive model.

### Key Takeaways

- **Data Collection**: Data was sourced from fbref.com, covering multiple seasons of EPL matches. A Python script utilizing BeautifulSoup was employed to scrape and compile this data.
  
- **Data Preprocessing**: Data quality and consistency were ensured through cleaning and standardization. Unnecessary columns were removed, and categorical features were one-hot encoded. Numerical features were adjusted to enhance model accuracy.

- **Feature Engineering**: Key features were selected based on their relevance to match outcomes, and rolling averages were used to capture temporal patterns in team performance.

- **Model Selection**: Random Forest and XGBoost models were chosen for their non-parametric nature, making them suitable for the nonlinear relationships in the data. Both models demonstrated strong performance, with XGBoost achieving a slightly higher accuracy.

- **Model Improvement**: Hyperparameter tuning via Grid Search and the introduction of rolling averages were implemented to enhance model performance. These adjustments helped in refining predictions and improving overall accuracy.

### Future Directions

While the current models provide valuable insights, there are opportunities for further development:

- **Dynamic Data Updates**: Incorporating batch processing to automatically update match data in real-time would keep the models current and improve prediction accuracy.
  
- **Advanced Techniques**: Exploring more sophisticated techniques such as ensemble methods or deep learning could potentially offer even better performance.

- **Additional Features**: Including more detailed features, such as player statistics or advanced team metrics, could enrich the model and improve its predictive power.

Overall, this project highlights the potential of machine learning in sports analytics and provides a foundation for future enhancements and applications in the field.

