import pandas as pd
from sklearn.decomposition import PCA

from ...utils import filename_grabber
from ...utils.config import settings
from ...utils.logger import get_logger


# Create logger
logger = get_logger(__name__)


def pca_analysis(df):
    """
    Applies pca to the given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to PCA.

    Returns:
        pandas.DataFrame: The PCA output DataFrame.
    """
    logger.debug("Applying PCA to data...")

    df_principal = df.drop(columns=['slug'])

    # Implement PCA analysis to reduce the number of features
    pca = PCA(n_components=None)
    principalComponents = pca.fit_transform(df_principal)
    df_principal = pd.DataFrame(data = principalComponents)

    # Add the slug column back to the DataFrame
    df_principal['slug'] = df['slug']
    

    return df_principal


# Use R for techniques on column reduction

