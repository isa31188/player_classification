import numpy as np
import pandas as pd
import warnings

def index_for_percentiles(array_length, num_splits):
    """
    Returns indices that split an array into num_split parts.
    Splits with larger size are the ones closer to the nth tile.
    Parameters:
    array_length (int): size of the array to split.
    param num_splits (int): number of splits, 100 for percentiles, 10 for deciles, etc.
    Return: 
    split_ind: [1, 1, 1,..., 1, 2, 2,2..., num_splits-1, num_splits-1, num_splits-1,num_splits, num_splits]
    """
    # if array_length cannot be split into equally large parts, some will have "normal" size while others will have +1 element
    delta = np.floor(array_length * 1. / num_splits) 
    normal_size = np.arange(1, delta + 1)
    larger_size = np.arange(1, delta + 2)

    # the number of "normal" splits that we can have
    times_large_split = array_length - delta * num_splits
    times_normal_split = num_splits - times_large_split
    
    # create indices and concatenate all
    ind_normal = np.array([np.arange(1,                      times_normal_split + 1) for i in normal_size]) # indices of normal-size splits
    ind_large  = np.array([np.arange(times_normal_split + 1,         num_splits + 1) for i in larger_size]) # indices of large-size splits
    
    ind_normal = (ind_normal.T).reshape((1, ind_normal.size))[0]
    ind_large  = (ind_large.T).reshape((1, ind_large.size))[0]
    all_ind = np.concatenate((ind_normal, ind_large))
    
    #check if all is ok and raise a warning if not
    if (len(all_ind) != array_length):
        warnings.warn("get_index_for_percentiles: PLEASE FIX ME. input lenght " + str(array_length) + " my length " + str(len(all_ind)))
    
    return np.sort(all_ind).round()

def quantile_metrics(y_true, y_prob, quantile_idxs):
    """
    Returns a dataframe with lift, max_lift, and precision for each quantile.
    Parameters:
    y_true (int array): array with true labels.
    y_prob (float array): array with predicted probabilities.
    quantile_idxs (int array): array with indices for splitting y_true and y_prob into quantiles.
    Return: 
    tile_counts: dataframe with n rows (e.g. 100 for percentiles) and columns lift, max_lift, precision.
    """
    
    mean_payers_per_tile = len(y_true[y_true == 1]) / np.max(quantile_idxs) # required for computing the lift
    
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    
    # quantiles computed according to prediced class probability
    df = df.sort_values(by='y_prob')
    df['quantile'] = quantile_idxs 
    quantile_metrics = df.groupby('quantile')['y_true'].agg(['count', 'sum'])
    quantile_metrics.columns = ['n', 'tp']
    # n = elements per quantile, tp = true positives per quantile.

    
    # Computing max lift in each percentile. 
    # Payers are firstly placed in higher percentiles. 
    # The max lift possible is the number of payers in a quantile divided by the average number of payers per quantile.
    max_lifts = df.sort_values(by='y_true')
    max_lifts['quantile'] = quantile_idxs 
    max_lifts_per_quantile = max_lifts.groupby('quantile')['y_true'].sum() / mean_payers_per_tile
    
    quantile_metrics['lift']      = quantile_metrics['tp'] / mean_payers_per_tile
    quantile_metrics['precision'] = quantile_metrics['tp'] / quantile_metrics['n'] # all elements in the quantile predicted positives.
    
    quantile_metrics.sort_index(inplace=True) # need to sort quantiles to be in the same order as the max_lifts_per_quantile array
    quantile_metrics['max_lift'] = max_lifts_per_quantile
    
    return quantile_metrics

def compare_model_quantile_metrics(quantile_metrics, model_names, n):
    """
    Returns dataframe comparing quantile metrics for different models.
    Parameters:
    quantile_metrics: array of dataframes containing quantile metrics computed by the function quantile_metrics for each model.
    model_names: array of model names for indexing the returned dataframe - ease of reading.
    n: rows at the top quantiles to use for comparing models.
    Return: 
    quantile_metrics_sample: dataframe with n*#models and columns quantile, count, tp, lift, precision, max_precision.
    """
    model_name_inds = np.arange(0, len(model_names))
    
    quantile_metrics_sample = pd.concat([quantile_metrics[idx].tail(n).reset_index() for idx in model_name_inds])
    quantile_metrics_sample.index = np.reshape([[model_names[idx]]*n for idx in model_name_inds], (len(model_names)*n,1)).T[0]
    
    return quantile_metrics_sample
