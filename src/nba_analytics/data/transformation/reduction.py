"""
This module provides functions for applying PCA (Principal Component Analysis) on NBA player statistics data.
The PCA technique is used to reduce the dimensionality of the dataset while retaining the most important features.
"""

import pandas as pd
from sklearn.decomposition import PCA

from ...utils import filename_grabber
from ...utils.config import settings
from ...utils.logger import get_logger

# Create logger
logger = get_logger(__name__)

def pca_analysis(df):
    """
    Applies Principal Component Analysis (PCA) to the given DataFrame to reduce the number of features.

    Args:
        df (pandas.DataFrame): The DataFrame to which PCA will be applied. 
                               The 'slug' column is excluded from PCA and added back to the resulting DataFrame.

    Returns:
        pandas.DataFrame: The DataFrame after applying PCA, with principal components and the 'slug' column.
    """
    logger.debug("Applying PCA to data...")

    # Drop the 'slug' column before applying PCA
    df_principal = df.drop(columns=['slug'])

    # Apply PCA analysis to reduce the number of features
    pca = PCA(n_components=None)
    principalComponents = pca.fit_transform(df_principal)
    df_principal = pd.DataFrame(data=principalComponents)

    # Add the 'slug' column back to the DataFrame
    df_principal['slug'] = df['slug']
    
    logger.debug("PCA applied successfully.")

    return df_principal

# Use R for techniques on column reduction
