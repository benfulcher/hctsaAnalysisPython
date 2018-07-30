#-------------------------------------------------------------------------------
# Visualize labeled HCTSA dataset as a UMAP projection
# Learn about umap here:
# http://umap-learn.readthedocs.io/en/latest/basic_usage.html
#-------------------------------------------------------------------------------
import matplotlib as plt
import seaborn as sns
import pandas as pd
import umap
import Matlab_IO
import sklearn
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

# Use OutputToCSV to generate the following (default) files:
matFilePath = '/Users/benfulcher/DropboxSydneyUni/CurrentProjects/Reduced_TimeSeriesStats/HCTSA_UCR_Results/HCTSA_AALTDChallenge.mat'
dataMatrixCSV = 'hctsa_datamatrix.csv'
timeSeriesInfoCSV = 'hctsa_timeseries-info.csv'

#-------------------------------------------------------------------------------
def LoadResults(fromCSV=True):
    """ Load relevant hctsa results as a dataframe:
    """
    if fromCSV:
        dataMatrix = pd.read_csv(dataMatrixCSV,header=None)
        tsLabels = pd.read_csv(timeSeriesInfoCSV,header=None,usecols=[0,1],names=('name','label'))
        tsLabels['label'] = tsLabels['label'].astype('category')
        return dataMatrix,tsLabels
    else:
        # Try using Matlab_IO
        retrieveThese = ('TimeSeries','TS_DataMat')
        loadedData = Matlab_IO.read_from_mat_file(matFilePath,retrieveThese)
        # loadedData[0]['keywords']

#-------------------------------------------------------------------------------
def UMAP_embed(dataMatrix):
    """ Compute a 2d umap projection of the data in dataMatrix
    """
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(dataMatrix)
    df = pd.DataFrame(data=embedding,columns=('umap-1','umap-2'))
    return df

def PCA_embed(dataMatrix):
    """ Compute a 2d PCA projection of the data in dataMatrix
    """
    pca = sklearn.decomposition.PCA(n_components=2)
    pca.fit(dataMatrix)
    embedding = pca.transform(dataMatrix)
    df = pd.DataFrame(data=embedding,columns=('PC-1','PC-2'))
    return df

#-------------------------------------------------------------------------------
def plot_projection(df,xData,yData,doSave=False):
    """ Plot two-dimensional projection of data
    """
    lowDim = sns.lmplot(x=xData, y=yData, data=df, fit_reg=False, markers='.',
                        hue='label', legend=True, legend_out=True, palette='Set2')
    # ax = plt.gca()
    # ax.set_title('UMAP projection of the hctsa dataset', fontsize=24);
    # ax.set_aspect('equal', 'datalim')
    if doSave:
        lowDim.savefig('umapProjection.pdf')
    return lowDim

#-------------------------------------------------------------------------------
def main():
    # Load data from hctsa calculation:
    dataMatrix,tsLabels = LoadResults()
    # Compute a umap projection in a dataframe:
    df_umap = UMAP_embed(dataMatrix)
    df_PCA = PCA_embed(dataMatrix)
    # Add label information:
    df_umap['label'] = tsLabels['label']
    df_PCA['label'] = tsLabels['label']

    # Plot and save:
    lowDim = plot_projection(df_umap,'umap-1','umap-2')
    fileName = 'umapProjection.png'
    lowDim.savefig(fileName)
    display("Saved output to %s" % fileName)

    # Plot and save:
    lowDim = plot_projection(df_PCA,'PC-1','PC-2')
    fileName = 'PCProjection.png'
    lowDim.savefig(fileName)
    display("Saved output to %s" % fileName)

if __name__ == "__main__": main()
