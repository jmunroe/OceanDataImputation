import os
import pickle
import pandas as pd
import numpy as np

class BaseImputer():
    def __init__(self, **params):
        # Store hyperparameters and parameters for this imputer
        
        self.name = 'BaseImputer'     

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return "BaseImputer()"
        
    def impute(self, df):
        # impute with this imputation algorithm
        
        df_imputed = df.copy()

        return df_imputed

from sklearn.impute import SimpleImputer

class MeanImputer(BaseImputer):
    def __init__(self, **params):
        # Apply hyperparameters and parameters for this imputer
        self.name = 'Mean'
        
    def __repr__(self):
        return "MeanImputer()"
        
    def impute(self, df):
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)

        return df_imputed

class LinearImputer(BaseImputer):
    def __init__(self):
        self.name = 'Linear'

    def __repr__(self):
        return "LinearImputer()"
        
    def impute(self, df):
        df_imputed = df.interpolate(method='linear')

        return df_imputed


from imputeMF import imputeMF

class MissForestImputer(BaseImputer):
    def __init__(self, max_iter=10):
        self.name = 'MissForest'

        self.max_iter = max_iter

    def __repr__(self):
        return f"MissForestImputer(max_iter={self.max_iter})"
        
    def impute(self, df):
        df_imputed = pd.DataFrame(imputeMF(df.values), columns=df.columns, index=df.index)

        return df_imputed

from sklearn.impute import KNNImputer as sklearnKNNImputer

class KNNImputer(BaseImputer):
    def __init__(self, n_neighbors=7):
        self.name = 'KNN'
        self.n_neighbors = n_neighbors

    def __repr__(self):
        return f"KNNImputer(n_neighbors={self.n_neighbors})"
        
    def impute(self, df):
        imputer = sklearnKNNImputer(n_neighbors=self.n_neighbors)       
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)

        return df_imputed

from statsmodels.imputation.mice import MICEData

class MICEImputer(BaseImputer):
    def __init__(self, k_pmm=3, n_iterations=10):
        self.name = 'MICE'
        
        self.k_pmm = k_pmm
        self.n_iterations = n_iterations
        
    def __repr__(self):
        return f"MICEImputer(k_pmm={self.k_pmm},n_iterations={self.n_iterations})"
        
    def impute(self, df):
        mice_data = MICEData(df, k_pmm=self.k_pmm)
        mice_data.update_all(self.n_iterations)

        df_imputed = pd.DataFrame(mice_data.data.values, columns=df.columns, index=df.index)

        return df_imputed

def add_artificial_gaps(df, num_sites=1, gap_length=30,
                       random_seed=None):
    """
    This function takes a dataframe and create contiguous gaps in place

    The random_seed is used to ensure reproducible results.
    Set to None of 
    Returns dictionary to indicate where these gaps have been added.
    """
    
    np.random.seed(random_seed)

    gaps = {}
    # randomly set a n-day contiguous segment as missing for each column
    random_columns = np.random.choice(df.columns, size=num_sites, replace=False)
    
    N = len(df.values.flatten())
    m = df.isnull().values.flatten().sum()
    missing_data = m / N * 100
    
    for col in random_columns:
        # Randomly select the start of the n-day segment
        start_idx = np.random.randint(0, len(df) - gap_length)
        end_idx = start_idx + gap_length
    
        gaps[col] = [start_idx, end_idx]
    
        # Set the values in this range to NaN
        df.iloc[start_idx:end_idx, df.columns.get_loc(col)] = np.nan

    m = df.isnull().values.flatten().sum()    
    missing_fraction = float(m / N * 100)

    return {'gaps': gaps, 'num_sites': num_sites, 'gap_length': gap_length, 'missing_fraction': missing_fraction}

import re

def make_valid_filename(s, replacement='_'):
    # Remove leading/trailing whitespace
    s = s.strip()
    # Replace invalid characters with the replacement
    s = re.sub(r'[\\/*?:"<>|]', replacement, s)
    # Optionally collapse multiple replacements
    s = re.sub(f'{re.escape(replacement)}+', replacement, s)
    return s

def run_experiment(imputer, minimum_missing_data=0, 
                   num_sites=5, gap_length=30, rerun_experiment=False,
                   dataset='dataset.csv', 
                   random_seed=4152):
    """
    This routine implements the steps
    
    1. Given a dataset, artificially introduce gaps to represent missing data.
    2. Using a specific imputatations method, impute those gaps.
    3. Measure the error between the orignal dataset and the imputed dataset.

    The results are cached in a pickle file and passed back as a dictionary.
    """

    experiment_name = f"{repr(imputer)}_missing{minimum_missing_data}_sites{num_sites}_gaplen{gap_length}"
    experiment_name = make_valid_filename(experiment_name)
    results_filename = f'results/{experiment_name}.pkl'

    if not rerun_experiment and os.path.exists(results_filename):
        with open(results_filename, 'rb') as f:
           results = pickle.load(f)
            
        return results
    
    # load the data
    df = pd.read_csv(dataset, parse_dates=True, index_col=0)
    df = df.rename(columns = lambda x: 'S_'+x)
      
    # Calculate the percentage of non-missing data for each study site
    non_missing_percentage = df.notna().mean() * 100
    
    # Filter study sites with at least 90% non-missing data
    selected_sites = non_missing_percentage[non_missing_percentage >= minimum_missing_data].index
    df_true = df[selected_sites].copy()

    # add artifical gaps 
    df = df_true.copy()
    results = add_artificial_gaps(df, num_sites, gap_length,
                                 random_seed=random_seed)

    # impute the missing data using the supplied method
    df_imputed = imputer.impute(df)

    # calculate metrics
    error = df_imputed - df_true
    MAE = np.mean(abs(error))
    RMSE = np.sqrt(np.mean((error)**2))

    # capture metadata and data from the experiment
    results.update( { 'experiment_name': experiment_name,
                      'imputer_name': str(imputer),
                      'imputer_details': repr(imputer),
                      'df_true': df_true,
                      'df' : df,
                      'df_imputed': df_imputed,
                      'RMSE' : RMSE,
                      'MAE' : MAE } )

    os.makedirs('results', exist_ok=True)
    with open(f'results/{experiment_name}.pkl', 'wb') as f:
        pickle.dump(results, f)
        
    return results
                    