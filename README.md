# Classify-MealNoMeal
 GIven time series data classify whether meal or no meal
 
 ## Models Used:-
 
 ### Random Forest
  Main intuition behind using Random Forest was that the time series data had a lot of features hence this classification method seemed convenient.
     
 ### SVM
   Since the probelm was a binary classification problem SVM was chosen. 
 
 ### Adaboost
   Adaboost was only chosen for training validation of other models.
     
 ## Scores:-
   Overall, after performing k-fold cross validation on all models, Random Forest performed the best to reach a peak of about 70% precision and about 69% accuracy/. Then it was SVM which was about the same range just shy of few numbers. 
     
## Features Selected:-
 * FFT (Fast-Fourier-Transform)
 * Entropy
 * Moving Standard Deviation
 * Kurtosis
 
 
