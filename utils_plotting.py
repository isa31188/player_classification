import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

def compute_and_plot_rocs(models, X_sets, y, colors, linestyles, ax, legend, title):
    """
    Computes and plots on the same graph the ROC curves for a set of models
    Parameters:
    models: array of n models with .best_estimator_
    X_sets: array of n datasets for computing the ROCs for each model
    colors: array of n colors ('green', 'blue', 'orange', ...)
    linestyles: array of n linestyles ('dashed', 'solid', ...)
    ax: the axis on which to plot. This assumes a command fig, ax = plt.subplots(...) was run before calling this function.
    legend: array of n strings for naming the different model curves
    title: string with plot title
    Return: 
    fpr: array of n arrays, each with false positive rates for each model and each threshold
    tpr: array of n arrays, each with true positive rates for each model and each threshold
    thresholds: array of n arrays, each with thresholds building the ROC curves for each model
    """
    fpr = {}; tpr = {}; thresholds = {}
    for idx, (model) in enumerate(models):
        fpr[idx], tpr[idx], thresholds[idx] = roc_curve(y, model.best_estimator_.predict_proba(X_sets[idx])[:,1])
        ax.plot(fpr[idx], tpr[idx], color=colors[idx], ls=linestyles[idx])
        ax.legend(legend)
        ax.set_title(title)
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')

    return fpr, tpr, thresholds


# plot pre-calculated confusion matrix with color and annotations
def plot_confusion_matrix(cm, target_names, ax, title='Confusion matrix', cmap=plt.cm.Blues, format='%d'):
    """
    Plots a confusion matrix with annotations
    Parameters:
    cm: pre-calculated confusion matrix
    target_names: array with the name of each target class
    ax: the axis for plotting
    title: plot title
    cmap: color map
    format: for printing the annotations. For floats, using '%.3f' could be useful
    Return: 
    im: plot object
    """
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)

    # Loop over data dimensions and create text annotations.
    for i in range(len(target_names)):
        for j in range(len(target_names)):
            ax.text(j, i, format % cm[i, j],ha="center", va="center", color="black")
            
    return im

# Computes and plots a confusion matriz in absolute or normalized counts
def compute_and_plot_cm(models, model_names, X_sets, y, normalize=False, format='%d'):
    """
    Computes and plots confusion matrices for multiple models
    Parameters:
    models: array of n models
    X_sets: array of n datasets 
    y: the target array
    normalize: whether to plot the cm's in absolute counts of % of total
    format: for printing the annotations. For normalize=True, using '%.3f' could be useful
    Return: 
    cm: array of n confusion matrices, one for each model
    """
    cm = {}
    target_names = [0, 1]
    tick_marks = np.arange(len(target_names))
    
    fig, axs=plt.subplots(1,len(models), figsize=(4*len(models),4))

    plt.setp(axs, xticks=tick_marks, yticks=tick_marks, 
             xticklabels=target_names, yticklabels=target_names, 
             xlabel='Pred label', ylabel='True label')

    for idx, (model, X) in enumerate(zip(models, X_sets)):
        y_pred = model.best_estimator_.predict(X)
        cm[idx]=confusion_matrix(y, y_pred)
        if normalize==True:  
            im_cm = plot_confusion_matrix(cm=cm[idx]/len(y), target_names=target_names, ax=axs[idx], title=model_names[idx], format=format)
        else:
            im_cm = plot_confusion_matrix(cm=cm[idx],        target_names=target_names, ax=axs[idx], title=model_names[idx], format=format)

    plt.tight_layout()
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im_cm, cax=cbar_ax)
    return cm

def plot_model_aucs(models, model_names, X_sets, y, num_format):
    for idx, (model, X) in enumerate(zip(models, X_sets)):
        print('%20s' % model_names[idx],    num_format % model.best_estimator_.score(X, y.astype(int)))
        
        
def lift_plotting(quantile_metrics, colors, linestyles, model_names, ax):

    for idx, (color, linestyle) in enumerate(zip(colors, linestyles)):
        ax.plot(quantile_metrics[idx].reset_index()['quantile'], quantile_metrics[idx]['lift'], color=color, ls=linestyle)

    ax.set_xlabel('Quantile'); ax.set_ylabel('Lift'); ax.set_title('Quantile lifts')
    ax.legend(model_names+['Max lift'])
    
    ax.plot(quantile_metrics[idx].reset_index()['quantile'], quantile_metrics[idx]['max_lift'], color='red')
