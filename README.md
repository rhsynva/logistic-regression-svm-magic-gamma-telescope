#Magic Gamma Telescope Classification: Logistic Regression and SVM Analysis
This project focuses on classifying high-energy gamma particles using the MAGIC Gamma Telescope dataset by implementing and comparing two distinct linear and non-linear classification models: Logistic Regression and Support Vector Machines (SVM). The goal is to simulate the registration of high-energy gamma particles in a ground-based atmospheric Cherenkov telescope. The classification distinguishes between Gamma rays (signal) and Hadron showers (background). By using Logistic Regression and SVM, this project evaluates how different mathematical approaches—probability-based vs. margin-based—handle the complex geometry of particle shower data.
The project was developed using Python in a Jupyter Notebook environment. I utilized Pandas and NumPy for data manipulation, Matplotlib and Seaborn for visualization, and Scikit-learn for machine learning tasks including LogisticRegression, SVC, GridSearchCV, and StandardScaler.

Data Preprocessing
To ensure the models receive clean data, several steps were performed. Redundant columns were dropped using the drop function to reduce dimensionality. For outlier detection and removal, I utilized Boxplots and Histograms to identify columns with a high probability of outliers. Based on the Boxplot analysis, these outliers were removed. In the visual representation, the empty circles represent outliers while the whiskers represent the borders. Any resulting empty spaces from the removal were filled with the mean of the respective column to maintain data integrity. Finally, Label Encoding was performed to prepare the target variable.

Training and Scaling
The dataset was divided into 80% Training and 20% Testing sets. Since both Logistic Regression and SVM are sensitive to the magnitude of feature values, Standard Scaling is mandatory. Scaling ensures that the regularization parameters and the hyperplane calculations are not biased by features with larger numerical ranges, allowing all features to contribute equally to the decision-making process.

Model Implementations
Logistic Regression Baseline
I first implemented Logistic Regression to establish a probabilistic baseline. Using Grid Search, I tuned the regularization strength to find the optimal balance between bias and variance. This model provides a probability score for each classification, offering insight into the statistical likelihood of a particle being a Gamma ray or a Hadron shower. After optimization, a Classification Report was generated to assess initial performance.

Support Vector Machines (SVM) Analysis
Following the logistic model, I implemented SVM using the same methodology as my previous KNN analysis. SVM seeks to find the optimal hyperplane that maximizes the margin between the two classes.

Final Evaluation and Comparison
The final stage of the project involved comparing the outcomes of both models through Classification Reports. I focused specifically on the F1-Score to ensure that both Precision and Recall were optimized. This comparative analysis demonstrates how non-linear kernels in SVM can often capture more complex patterns in particle data compared to the linear assumptions of standard Logistic Regression.
