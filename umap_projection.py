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

#-------------------------------------------------------------------------------
# Use OutputToCSV to generate the following (default) files:
# matFilePath = '/Users/benfulcher/DropboxSydneyUni/CurrentProjects/Reduced_TimeSeriesStats/HCTSA_UCR_Results/HCTSA_AALTDChallenge.mat'
dataMatrixCSV = 'hctsa_datamatrix.csv'
timeSeriesInfoCSV = 'hctsa_timeseries-customGroup.csv'

#-------------------------------------------------------------------------------
# SNS plot style settings:
# sns.set(style='white', context='notebook', rc={'figure.figsize':(7,5)})
# sns.set(rc={'figure.figsize':(6,4),"font.size":16,"axes.titlesize":18,"axes.labelsize":14},style="white")
# sns.set(rc={'figure.figsize':(6,4),"font.size":16,"axes.titlesize":18,"axes.labelsize":14},style="white")

#-------------------------------------------------------------------------------
def LoadResults(fromCSV=True):
    "Load relevant hctsa results as a dataframe"
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
    "Compute a 2d umap projection of the data in dataMatrix"
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(dataMatrix)
    df = pd.DataFrame(data=embedding,columns=('umap-1','umap-2'))
    return df
#-------------------------------------------------------------------------------
def tSNE_embed(dataMatrix):
    "Compute a 2d t-SNE projection of the data in dataMatrix"
    reducer = sklearn.manifold.TSNE(n_components=2)
    embedding = reducer.fit_transform(dataMatrix)
    df = pd.DataFrame(data=embedding,columns=('tSNE-1','tSNE-2'),index=dataMatrix.index)
    return df
#-------------------------------------------------------------------------------
def PCA_embed(dataMatrix):
    "Compute a 2d PCA projection of the data in dataMatrix"
    pca = sklearn.decomposition.PCA(n_components=2)
    pca.fit(dataMatrix)
    embedding = pca.transform(dataMatrix)
    df = pd.DataFrame(data=embedding,columns=('PC-1','PC-2'))
    return df
#-------------------------------------------------------------------------------
def plot_projection(df,xData,yData,doSave=False,customPalette='Paired',showLegend=True):
    "Plot two-dimensional projection of data"
    lowDim = sns.lmplot(x=xData,y=yData,data=df,fit_reg=False,markers='o',
                        hue='label',legend=showLegend,legend_out=showLegend,palette=customPalette)
    # palette='seismic'
    # ax = plt.gca()
    # ax.set_title('UMAP projection of the hctsa dataset', fontsize=24);
    # ax.set_aspect('equal', 'datalim')
    if doSave:
        lowDim.savefig('umapProjection.pdf')
    return lowDim

#-------------------------------------------------------------------------------
# def main():
# Load data from hctsa calculation:
dataMatrix,tsLabels = LoadResults()

# Compute low-dimensional projections:
df_umap = UMAP_embed(dataMatrix)
df_PCA = PCA_embed(dataMatrix)
df_tSNE = tSNE_embed(dataMatrix)

# Add label information:
df_umap['label'] = tsLabels['label']
df_PCA['label'] = tsLabels['label']
df_tSNE['label'] = tsLabels['label']

#-------------------------------------------------------------------------------
# Plotting options:

# Select a color palette for the data:
pairedPalette = sns.color_palette('Paired',8)
# myOrder = [1,0,3,5,4,2] # Excitatory_SHAM
# myOrder = [3,2,0,1,4,5,6,7] # Excitatory_PVCre_SHAM
# myOrder = [3,2,5,4] # Excitatory_PVCre
myOrder = [3,2,1,0] # PVCre_SHAM
myPalette = [pairedPalette[i] for i in myOrder]
# myPalette = sns.color_palette("hls")

# Set output style:
sns.set_context("notebook",font_scale=1.7)

#-------------------------------------------------------------------------------
# Plot and save (UMAP):
lowDim = plot_projection(df_umap,'umap-1','umap-2',customPalette=myPalette)
fileName = 'umapProjection.svg'
lowDim.savefig(fileName)
display("Saved output to %s" % fileName)

# Plot and save (t-SNE):
lowDim = plot_projection(df_tSNE,'tSNE-1','tSNE-2',customPalette=myPalette)
fileName = 'tSNEProjection.svg'
lowDim.savefig(fileName)
display("Saved output to %s" % fileName)

# Plot and save:
lowDim = plot_projection(df_PCA,'PC-1','PC-2',customPalette=myPalette,showLegend=False)
fileName = 'PCProjection.svg'
lowDim.savefig(fileName,bbox_inches='tight')
display("Saved output to %s" % fileName)

# if __name__ == "__main__": main()
