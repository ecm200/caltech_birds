import pandas as pd

def get_manifold_df(trf_data, labels_truth, labels_pred, class_names, img_paths):
    ''' 
    Function to build a dataframe from the TSNE results, and append with other information such as:

        0. TSNE components.
        
        1. Labels data (truth and predicted).

        2. Class names (truth and predicted).

        3. Path to the images.
        
    '''
    
    class_truth = []
    class_pred = []
    for label_truth in labels_truth:
        class_truth.append( class_names[label_truth] )
        
    for label_pred in labels_pred:
        class_pred.append( class_names[label_pred] )
        
    return pd.DataFrame({'Manifold Dim 1' : trf_data[:,0], 
                         'Manifold Dim 2' : trf_data[:,1], 
                         'label (truth)' : labels_truth,
                         'label (pred)' : labels_pred,
                         'class name (truth)' : class_truth,
                         'class name (pred)' : class_pred,
                         'image path' : img_paths})