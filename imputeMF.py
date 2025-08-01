import numpy as np
from sklearn.ensemble import RandomForestRegressor

def imputeMF(X, max_iter=10, print_stats=False):

    mask = np.isnan(X)

    # Count missing per column
    col_missing_count = mask.sum(axis=0)
    
    # Get col and row indices for missing
    missing_rows, missing_cols = np.where(mask)

    Ximp = X.copy()

    # 1. Make initial guess for missing values
    col_means = np.nanmean(X, axis=0)
    Ximp[missing_rows, missing_cols] = np.take(col_means, missing_cols)

    # 2. misscount_idx: sorted indices of cols in X based on missing count
    misscount_idx = np.argsort(col_missing_count)

    iter_count = 0
    gamma_new = 0
    gamma_old = np.inf

    col_index = np.arange(Ximp.shape[1])

    rf_regressor = RandomForestRegressor(n_jobs=-1)

    while(gamma_new < gamma_old and iter_count < max_iter):
        
        #4. store previously imputed matrix
        Ximp_old = np.copy(Ximp)
        
        if iter_count != 0:
            gamma_old = gamma_new
            
        #5. loop for s in k
        for s in misscount_idx:
            # Column indices other than the one being imputed
            s_prime = np.delete(col_index, s)
        
            # Get indices of rows where 's' is observed and missing
            obs_rows = np.where(~mask[:, s])[0]
            mis_rows = np.where(mask[:, s])[0]
        
            # If no missing, then skip
            if len(mis_rows) == 0:
                continue
        
            # Get observed values of 's'
            yobs = Ximp[obs_rows, s]
        
            # Get 'X' for both observed and missing 's' column
            xobs = Ximp[np.ix_(obs_rows, s_prime)]
            xmis = Ximp[np.ix_(mis_rows, s_prime)]
        
            # 6. Fit a random forest over observed and predict the missing
            rf_regressor.fit(X=xobs, y=yobs)
            
            # 7. predict ymis(s) using xmis(x)
            ymis = rf_regressor.predict(xmis)
            
            # 8. update imputed matrix using predicted matrix ymis(s)
            Ximp[mis_rows, s] = ymis
        
        # 9. Update gamma (stopping criterion)
        gamma_new = np.sum((Ximp - Ximp_old) ** 2) / np.sum(Ximp ** 2)
        iter_count += 1

        if print_stats:
            print('Statistics:')
            print(f'iteration {iter_count}, gamma = {gamma_new}')

    return Ximp