##ALL THE FUNCTIONS NEEDED FOR THE PIPELINE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FuncFormatter
import seaborn as sns
from scipy.stats import hypergeom
from scipy.spatial.distance import jensenshannon
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import umap
import igraph as ig
import leidenalg as la
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D 

pd.set_option("display.max_columns", None)  # None = no limit for when i do df.head()


##LIST CONTAINITNG THE GROUPS NAMES 
treatment_groups = [
        'BT474_mV_72hNHWD',
        'BT474_mV_72hSTC15',
        'BT474_mV_high_Untreated',
        'BT474_mV_low_Untreated',
        'BT474_mV_Untreated_Unsorted']

## ARBITRARY CHOSEN COLORS FOR THE CLUSTERS (THEY NEED TO HAVE A HIGH VALUE AND CHROMA TO STAND OUT ON A GREY BACKGROUND)
hex_colors = [
    "#0e67a7", "#ff7f0e", "#a0e468", "#d62728", "#9467bd",
    "#672417", "#e377c2", "#f5f523", "#28e0f5", "#3214a8",
    "#ca9d16", "#04a887"
]

#MANUALLY RECALCULATING POL2 POSITION AS THE MIDPOINT BETWEEN MOTIF_START AND MOTIF_END
def calculate_pol2_position(df):
    """Calculate the Pol II position as the midpoint between motif_start and motif_end."""
    if 'motif_start' in df.columns and 'motif_end' in df.columns:
        df['pol2_pos']= (df['motif_start'] + df['motif_end']) / 2
    else:
        raise ValueError("DataFrame must contain 'motif_start' and 'motif_end' columns.")
    return df

## PREPROCESS OF THE DATAFRAME FUNCTION

def preprocess_dataframe(
    df: pd.DataFrame,
    #nan_threshold: float = 0.9,
    drop_columns: list = None,
) -> pd.DataFrame:
    
    """
    Preprocess a DataFrame by:
    # Dropping columns with too many NaN values
    0. Creating the 'pol2_pos' column as the midpoint between 'motif_start' and 'motif_end' (ignorinng 'motif_center' column, in case error in dataset)
    1. Removing user-defined non-useful columns
    2. Converting 'chr' column to numeric
    3. Converting the variable 'pol2' into a binary marker: 1 for pol2 and 0 for no pol2
    4. Creating a unique 'gene_tss' identifier
    5. Dropping duplicate rows
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to preprocess.
    #nan_threshold : float, optional (default=0.9) (each column that has more than 10% of NaN is dropped)
        Minimum fraction of non-NaN values required to keep a column.
    drop_columns : list, optional
        Additional columns to drop.
    
    Returns
    -------
    pd.DataFrame
        Cleaned and preprocessed DataFrame.
    """

    # Work on a copy to avoid modifying original
    df_proc = df.copy().drop_duplicates()

    # # Drop columns with too many NaN values
    # coverage_mask = df_proc.notna().mean(axis=0) > nan_threshold    #dropping columns that have too many NaN
    # dropped_columns = df_proc.columns[~coverage_mask].tolist()      #lst of dropped columns
    # df_proc = df_proc.loc[:, coverage_mask]                         #new df only keeping useful columns
    # print(f"Filtered from {df.shape[1]} to {df_proc.shape[1]} regions") #tells how many columns dropped
    # print("Dropped (NaN coverage):", dropped_columns)                   #lists the dropped columns

    # Create 'pol2_pos' column as midpoint between 'motif_start' and 'motif_end'
    df_proc = calculate_pol2_position(df_proc)

    # Drop user-deemed non-useful columns
    if drop_columns:
        df_proc = df_proc.drop(columns=[col for col in drop_columns if col in df_proc.columns], errors="ignore")

    # Convert chr column to numeric (strip "chr")
    if "chr" in df_proc.columns:
        df_proc["chr"] = pd.to_numeric(df_proc["chr"].str.replace("chr", "", regex=False), errors="coerce")

    #Convert 'pol2' label into a binary marker (change pol2 into 1)
    if 'pol2' in df_proc.columns:
        mapping_pol2 = {"pol2":1, "nonpol2":0}
        df_proc["pol2"] = df_proc["pol2"].map(mapping_pol2)

    # Create gene_tss column if columns exist
    if "gene" in df_proc.columns and "tss_pos" in df_proc.columns:
        df_proc["gene_tss"] = df_proc["gene"].astype(str) + "_" + df_proc["tss_pos"].astype(str)

    # Drop duplicates again after transformations
    df_proc = df_proc.drop_duplicates()

    print(f"Final shape: {df_proc.shape}")
    print(f"NUmber of gene_tss: {df_proc['gene_tss'].nunique() if 'gene_tss' in df_proc.columns else 'N/A'}")

    return df_proc


##FUNCTION TO FILTER ONLY THE GENES THAT HAVE SUFFICIENT READS THAT OVERLAP ON THE SAME GENOMIC REGION, CENTERED ON THE MIDDLE
## THE INPUT DF MUST BE BINNED

def filter_reads_per_gene_middle_bin_name(
    df: pd.DataFrame, #usually need to put the preprocessed dataframe BUT BINNED, with all the genes in there
    middle_bin_name: str =None,
    min_reads: int =50, # the minimal number of reads that we want
    min_bins: int =40,  # the miniml lenght per bin that we want the read to be
    require_middle_bin: bool =True
    
):
    """
    Filters reads per gene based on coverage in consecutive bins and keeps intervals
    containing a specified middle bin column.


    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with reads as rows, bins as columns, and a gene identifier column or index.
    middle_bin_name : str
        Name of the middle bin column to require in the kept intervals.
    min_reads : int
        Minimum number of overlapping reads per bin.
    min_bins : int
        Minimum number of consecutive bins required.
    require_middle_bin : bool
        If True, only keeps regions that contain the middle bin.

    Returns
    -------
    df_filtered : pd.DataFrame
        Reads from genes that pass the filter, with original indices preserved.
        THE DF IS BINNED AND IN MATRIX FORM WITH THE NAN VALUES
        IT IS GROUPED BY GENE, had had as index; 'gene-tss', 'readid', 'group' 'cluster
    filtered_regions : dict
        Per-gene list of intervals (start_bin, end_bin) that pass the filter.
    """
    
    # Detect bin columns
    if 'readid' in df.columns:   
        bin_cols = df.columns.difference(['readid']) #it creates a dataframe that keeps every other column of df except the columns 'readid'
        #it's just to take th bin columns, not the 'readid' column
    
    else:
        bin_cols = df.columns
    
    # Get index of middle bin column if specified
    if middle_bin_name is not None:
        if middle_bin_name not in bin_cols:
            raise ValueError(f"Middle bin '{middle_bin_name}' not found in bin columns")
        middle_bin_idx = bin_cols.get_loc(middle_bin_name)
    
    def find_high_coverage_regions(
            coverage, 
            min_reads, 
            min_bins
            ): 
        """
        Identifies continuous genomic regions (bins) where the coverage (number of overlapping reads) 
        is consistently above a minimum threshold, for at least a minimum region length.

        Parameters:
        -----
        coverage: list or np.ndarray
            A sequence of coverage values (e.g., per genomic bin). Each value indicates how many reads cover that bin.
        min_reads: int)
            The minimum number of reads required for a bin to be considered "covered."
        min_bins: int
            The minimum number of consecutive bins that must satisfy the coverage threshold in order to define a valid region.

        Returns:
        -----

        regions: list of tuples (start, end)
            Each tuple (start, end) represents the indices of bins forming a valid high-coverage region.
        """
        regions = []
        start = None
        for i, cov in enumerate(coverage):
            if cov >= min_reads:
                if start is None:
                    start = i
            else:
                if start is not None:
                    end = i - 1
                    if (end - start + 1) >= min_bins:
                        regions.append((start, end))
                    start = None
        if start is not None:
            end = len(coverage) - 1
            if (end - start + 1) >= min_bins:
                regions.append((start, end))
        return regions
    
    filtered_rows = []
    filtered_regions = {}

    # Group by gene
    groups = df.groupby(level='gene_tss', observed=True)
    
    for gene, sub in groups:
        coverage = sub[bin_cols].notna().sum(axis=0)
        regions = find_high_coverage_regions(coverage, min_reads, min_bins)
        
        # Keep only intervals covering the middle bin if required
        if require_middle_bin and middle_bin_name is not None:
            regions = [(s,e) for s,e in regions if s <= middle_bin_idx <= e]
        
        if not regions:
            continue  # skip gene if no valid region
        
        filtered_regions[gene] = regions
        
        # Keep reads overlapping at least one valid region
        mask = np.zeros(sub.shape[0], dtype=bool)
        for start, end in regions:
            mask |= sub[bin_cols].iloc[:, start:end+1].notna().any(axis=1)
        
        filtered_rows.append(sub[mask])
    
    if not filtered_rows:
        return pd.DataFrame(), {}
    
    df_filtered = pd.concat(filtered_rows, ignore_index=False)

    
    return df_filtered, filtered_regions



##BINNING THE DATAFRAME AND TURNING IT INTO MATRIX FORM
# TO BE CALLED IN A FUNCITON, NO USE ALONE

def bin_then_matrix(
        df : pd.DataFrame ,
        indexes : list = ['gene_tss','readid','group','cluster'], 
        bin_size : int =50,
):
        """
        Bins the reads, and then transforms the dataframe into a dataframe matrix with methylation values per bin

        Parameters:
        df : pd.dataframe
                the dataframe must have 'readid','gene_tss','group','cluster','meth', 'C_pos' as minimal columns
        
        indexes: list
                by default the following columns turn into indexes ['gene_tss','readid','group','cluster']
        
        bin_size : int
                by default 50 bp
        
        Returns:
        df_matrix: dataframe
                a matric that still is a dataframe, having as indexes gene_tss, readid, group, cluster, 
                and the bins as columns, and as value the mean methylation value per bin
        """

        #Creating a new 'bin' column:
        if 'C_pos' in df.columns:
                df['bin'] = (df['C_pos']//bin_size)*bin_size


        #Pivot the dataframe into a matrix (but it's still a dataframe)
        if 'readid' and 'group' in df.columns:
                df = df.pivot_table(index=indexes, columns='bin', values='meth', aggfunc='mean')
        
        return df


## PROCESSING -- MARKING THE METHYLATION OF READS BY BINNED REGIONS -- GIVING OUT A DATAFRAME

def process_into_matrix(
        df: pd.DataFrame,
        gene: object=None,          #the id taken from the column gene_tss, only if we want to process per gene
        bin_size: int = 50,         #bin size, by default 50bp
):

    """
    Process the DataFrame to turn it into a binned methylation value matrix:
    CAN EITHER BE USED ON THE WHOLE DATAFRAME WITH ALL THE GENES, OR ON A SINGLE GENE

    1. Binning on the GpC site position, relative to the Pol2 summit, by default bin size = 50bp
    2. Making a dataframe 'matrix' having as indexes: 'readid','group' and 'cluster' if processed per gene, adding 'gene_tss' if processed entirely
       as colums: 'bin', and as values: 'meth' (mean methylation value per bin)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input preprocessed dataframe 
    gene : object 
        Name of the gene, that we want to isolate and work on, BY DEFAULT SSUMING THAT WE WORK ON THE WHOLE DATAFRAME
    bin_size : integer
        By default set as 50 bp
    
    Returns
    -------
     
    pd.DataFrame
        with missing values, not yet a standardized matrix, ready to be processed in a PCA, and for treatments.
    """

    if gene != None:
        #filtering out to get the new dataframe to work on
        df=df[df['gene_tss']== gene]

        #Binning and then turning into a methylation matrix
        df= bin_then_matrix(df, indexes= ['readid','group','cluster'], bin_size= bin_size)

    else:
        df = bin_then_matrix(df, indexes= ['gene_tss','readid','group','cluster'], bin_size= bin_size)

     
    print(f"Shape of the dataframe : {df.shape}")
    
    return df


## TO HANDLE THE NAN VALUES ONCE IN MATRIX FORM

def handling_NaN(
        df: pd.DataFrame, # must be binned and in matrix form
        nan_threshold: float = 0.7, #limit to drop bins that have too many missing values
        nan_method: object = None 

):
        """
        Parameters:

        nan_threshold: float
        threshold for dropping bins that have too many missing values
        nan_method : string
        Choosing the method on how to deal with the missing methylation values, either by 'drop' or by 'impute', or eavint the NaN there. By default leaving them there

        Returns:
        df: the dataframe matrix with or without missing values depending on the nan method

        """
        #Dropping the reads that contain NaN values
        if nan_method == 'drop':
                
                # Drop bins with too many NaN values
                coverage_mask = df.notna().mean(axis=0) > nan_threshold   #dropping columns that have too many NaN
                dropped_columns = df.columns[~coverage_mask].tolist()      #lst of dropped columns
                df = df.loc[:, coverage_mask]                         #new df only keeping useful columns

                print("Dropped (NaN coverage):", dropped_columns)                   #lists the dropped columns

                i,j = df.shape
                #filter by dropping the reads that have too many NaN
                df = df.dropna(thresh=j)       # Keep rows with at least j non-NaN values (= dropping all rows that had a NaN)
        
        #Imputing the values with the kNN method
        elif nan_method == 'impute':
                
                # Drop bins with too many NaN values
                coverage_mask = df.notna().mean(axis=0) > nan_threshold   #dropping columns that have too many NaN
                dropped_columns = df.columns[~coverage_mask].tolist()      #lst of dropped columns
                df = df.loc[:, coverage_mask]                         #new df only keeping useful columns

                print("Dropped (NaN coverage):", dropped_columns)                   #lists the dropped columns

                imputer = KNNImputer(n_neighbors=5)
                X = imputer.fit_transform(df) #X_impute is a numpy array
                df.loc[:, :] = X        # Put results back into the same DataFrame

        #leaving the missing values otherwise:
        
        return df


def preprocess_long_for_plot(df, include_locus_cluster: bool = False):
    """
    Prepare dataframe in long format for plotting reads.
    Each row = one CpG per read.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns at least ['gene_tss', 'group', 'cluster', 'pol2_pos', "readid", "C_start", "meth", 'locus_cluster'].
    include_locus_cluster : bool, optional
        If True, keep 'locus_cluster' column (for plotting after clustering).
        If False, ignore it.
    """
    df = df.reset_index(drop=True)  # ensure features are columns
    
    # Define base columns
    base_cols = ["gene_tss", "group", "readid", "cluster", "C_start", "meth",'pol2_pos']
    keep_cols = [c for c in df.columns if c in base_cols]

    # Optionally add locus_cluster if it exists
    if include_locus_cluster and "locus_cluster" in df.columns:
        keep_cols.append("locus_cluster")

    df = df[keep_cols]

    # Compute per-read start/end (no lists!)
    span = (
        df.groupby("readid")["C_start"]
        .agg(["min", "max"])
        .rename(columns={"min": "read_start", "max": "read_end"})
    )
    df = df.merge(span, on="readid", how="left")

    return df


def plot_reads_long(
    df,
    filters=None,
    facet_by=None,
    color_by=None,
    hex_colors=None,
    max_facets=14
):
    """
    Plot per-read methylation (long format, one row per CpG per read), centered on Pol2 position.
    Facet heights scale with number of reads per facet.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe (one row per CpG per read).
        Must have 'read_start', 'read_end', and 'pol2_pos' columns.
    filters : dict, optional
        Column:value filters, e.g. {"gene_tss": "BRCA1", "group": "control"}.
    facet_by : str or list, optional
        Column(s) to facet subplots by (e.g. "group", ["gene_tss","group"]).
    color_by : str, optional
        Column to color read spans by (e.g. "cluster").
    hex_colors : list, optional
        List of hex colors to use for categories in `color_by`.
    max_facets : int
        Prevents generating too many subplots at once.
    """
    df = df.copy()

    # --- Filtering ---
    if filters:
        for col, val in filters.items():
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe.")
            if isinstance(val, list):
                df = df[df[col].isin(val)]
            else:
                df = df[df[col] == val]

    if df.empty:
        raise ValueError("No reads left after filtering!")

    # --- Faceting ---
    if facet_by is None:
        facet_values = [("All", df)]
    else:
        if isinstance(facet_by, str):
            facet_by = [facet_by]
        facet_values = list(df.groupby(facet_by))
        if len(facet_values) > max_facets:
            raise ValueError(f"Too many facets ({len(facet_values)}). Max allowed: {max_facets}")

    # --- Color mapping ---
    color_map = None
    if color_by:
        if color_by not in df.columns:
            raise ValueError(f"Column '{color_by}' not found in dataframe.")
        categories = sorted(df[color_by].dropna().unique())
        if hex_colors is None:
            import matplotlib.cm as cm
            cmap = cm.get_cmap("tab20", len(categories))
            hex_colors = [cmap(i) for i in range(len(categories))]
        if len(hex_colors) < len(categories):
            raise ValueError(f"Not enough colors for {len(categories)} categories.")
        color_map = dict(zip(categories, hex_colors))

    # --- Figure setup with gridspec ---
    n_facets = len(facet_values)
    heights = [max(1, subdf["readid"].nunique() * 0.15) for _, subdf in facet_values]  # scale heights
    total_height = sum(heights) + 2  # add some padding
    fig = plt.figure(figsize=(30, total_height))
    gs = gridspec.GridSpec(n_facets, 1, height_ratios=heights)

    # --- Plot each facet ---
    for i, (facet_key, subdf) in enumerate(facet_values):
        ax = fig.add_subplot(gs[i, 0])
        ax.set_facecolor("#919191")  # light gray background

        if "pol2_pos" not in subdf.columns:
            raise ValueError("Column 'pol2_pos' not found for centering!")

        # Shift positions relative to pol2_pos
        subdf = subdf.copy()
        subdf["C_start_shifted"] = subdf["C_start"] - subdf["pol2_pos"]
        subdf["read_start_shifted"] = subdf["read_start"] - subdf["pol2_pos"]
        subdf["read_end_shifted"] = subdf["read_end"] - subdf["pol2_pos"]

        # Order reads by color_by if requested
        if color_by and color_by in subdf.columns:
            grouped_reads = subdf.groupby(color_by)["readid"].unique().to_dict()
            read_order = [(cat, rid) for cat, rids in grouped_reads.items() for rid in rids]
        else:
            read_order = [(None, rid) for rid in subdf["readid"].unique()]

        # Plot each read
        for idx, (cat_value, readid) in enumerate(read_order):
            sub = subdf[subdf["readid"] == readid]
            span_color = color_map.get(cat_value, "black") if color_by else "black"

            ax.hlines(
                idx,
                xmin=sub["read_start_shifted"].iloc[0],
                xmax=sub["read_end_shifted"].iloc[0],
                color=span_color,
                linewidth=1.6
            )

            # CpG dots
            for _, row in sub.iterrows():
                dot_color = "white" if row["meth"] == 1 else "black"
                ax.plot(row["C_start_shifted"], idx, "o", color=dot_color, markersize=4)

        # Vertical line at Pol2
        ax.axvline(x=0, color="#C80028", linestyle="-", linewidth=2, label="Pol2 position")

        # Labels, limits, formatting
        ax.set_xlabel("Position relative to Pol2 (bp)", fontsize=12)
        ax.set_ylabel("Read IDs", fontsize=12)
        x_min = subdf["read_start_shifted"].min() - 100
        x_max = subdf["read_end_shifted"].max() + 100
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-2, len(read_order) + 2)

        # Facet title
        if facet_by:
            if isinstance(facet_key, tuple):
                title = ", ".join([f"{col}={val}" for col, val in zip(facet_by, facet_key)])
            else:
                title = f"{facet_by[0]}={facet_key}"
        else:
            title = "All Reads"
        ax.set_title(f"Read-level methylation centered on Pol2 ({title})", fontsize=16)
        ax.grid(True)
        ax.invert_yaxis()

    # --- Legend ---
    handles = [
        mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=6, label="Unmethylated (0)"),
        mlines.Line2D([], [], color="black", marker="o", markerfacecolor="white", linestyle="None", markersize=6, label="Methylated (1)"),
        mlines.Line2D([], [], color="#C80028", linestyle="-", linewidth=2, label="Pol2 position")
    ]
    if color_map:
        for cat, col in color_map.items():
            handles.append(mlines.Line2D([], [], color=col, linewidth=2, label=f"{color_by}={cat}"))

    fig.legend(handles=handles, loc="upper right", fontsize=9, frameon=True)
    fig.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.05)

    return fig


## SMALL FUNCTION TO GET THE LIST OF ALL THE GENES NAMES OF A DATAFRAME AND THE LIST OF ALL THE SUBDATAFRAME OF THESE GENES

def get_genes_list(df):
    gene_list_names= [] #the list that will take in the genes names
    gene_list_df=[] #the list that will take in the dataframes associated to the genes

    for gene, sub_df in df.groupby(df.index.get_level_values("gene_tss")):
        gene_list_names.append(gene)
        gene_list_df.append(sub_df)

    return gene_list_names,gene_list_df

## CLUSTERING ALOGRITHN FOLLOWING THE STEPS: PCA --> KNN GRAPH --> LEIDEN

def clustering(
        df: pd.DataFrame,       
        n_neighbors: int,   #to choose how many neighors for the kNN graph construction, before using leiden 
        nan_threshold : float = 0.7, #to drop the bins that have too many nan
        nan_method : str = 'drop', #by default dropping the rows that contain nan, but can use 'impute' too
        knn_metric: str = 'euclidean',   # the metric used for computing the kNN graph construction. Can also be cosine for ex
        leiden_resolution: float = 1.0 # the resolution of the leiden (small = less clusters)   
):
    """
    Clustering algorithm
    1. Handles the Nan values from the matrix-like dataframe
    2. Converts the dataframe into a numpy array
    3. Standardizes the values
    4. Runs the PCA
    5. creates a kNN graph
    6. runs Leiden on the graph

    Parameters:
    df: is a dataframe already binned and matrix-like, but simply has missing values in it, still,
    n_neighbors: int: for the knn graph construction
    nan_threshold: float: to drop the bins that have too many nan values
    knn_metric: str metric used for knn graph construction
    leiden_resolution: float 

    """
    #Handling the NaN:
    df= handling_NaN(df, nan_threshold, nan_method)
    
    #Converting into a np.array:
    X = df.to_numpy()
    
    # Standardization of the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # kNN graph construction
    knn = NearestNeighbors(n_neighbors= n_neighbors, metric= knn_metric)
    knn.fit(X_pca)
    knn_distances, knn_indices = knn.kneighbors(X_pca)

    # Create igraph (might change this function later with a manual coded function to create igraph)
    edges = [(i, j) for i in range(knn_indices.shape[0]) for j in knn_indices[i, 1:]]
    g= ig.Graph(edges=edges, directed=False)

    # Run Leiden 
    partition= la.find_partition(g, la.RBConfigurationVertexPartition, resolution_parameter= leiden_resolution, seed=42)   
    clusters= np.array(partition.membership)

    return df, X_pca, partition, clusters


## SCATTER PLOT FUNCTION : VISUALIZATION OF THE CLUSTERING BASED ON UMAP FOR A SINGLE GENE (MADE FOR THE STUDY OF A PARTICULAR GENE)

def plotting_the_clustering(
        X: np.array, # can either be the matrix after the PCA was ran, so X_PCA, or the raw X. Usually, it's the one after the PCA is ran. (that's what scanpy does)
        clusters : np.array, #should be the clusters given after running the leiden
        n_neighbors: int = 15, #not necessarily the same number of neighbors as in the kNN graph computing (high value loses local structure)
        min_dist: float = 0.1, # low value = more tightly packed embeddings
        metric: str = 'euclidean' #should be the same as the one used in the knn graph

):
    reducer = umap.UMAP(n_neighbors= n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
    embedding = reducer.fit_transform(X)
    
    unique_clusters = np.unique(clusters)
    
    cluster_colors = { cluster_id: hex_colors[i % len(hex_colors)] for i, cluster_id in enumerate(unique_clusters)}

    colors = [cluster_colors[c] for c in clusters]

    fig= plt.figure(figsize=(12,8))
    plt.scatter(embedding[:,0],
            embedding[:,1],
            c=colors,
            alpha=0.8)
    
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("UMAP for X")
    plt.grid(True)
    
    return fig


## MULTIPLE SCATTER PLOTS : TO VISUALIZE THE CLUSTERING BASED ON UMAP OF ALL THE GENES AT THE SAME TIME

def plot_multiple_umaps(df, genes_list, titles=None, leiden_neighbors= 15, umap_neighbors=15, resolution= 1.0, min_dist=0.1, leiden_metric = 'euclidean', umap_metric="euclidean"):
    """
    Plot multiple UMAP scatter plots in a checkerboard layout.
    
    Parameters
    ----------
    df: df_filtered, that is matrx like

    genes_list : list of array-like
        List of genes label arrays (same length as X).
    titles : list of str
        Optional titles for each subplot.
    n_neighbors, min_dist, metric : UMAP parameters
    """

    # Checkerboard size
    n_plots = len(genes_list)
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = axes.flatten()

    for i, genes in enumerate(genes_list):
        # 
        df_gene, X, partition, clusters = clustering(gene_df[i], leiden_neighbors, 0.7, 'drop', leiden_metric, resolution)

        # Compute shared UMAP embedding
        reducer = umap.UMAP(n_neighbors= umap_neighbors, min_dist=min_dist, metric=umap_metric, random_state=42)
        embedding = reducer.fit_transform(X)

        unique_clusters = np.unique(clusters)
    
        cluster_colors = { cluster_id: hex_colors[i % len(hex_colors)] for i, cluster_id in enumerate(unique_clusters)}

        colors = [cluster_colors[c] for c in clusters]

        sc = axes[i].scatter(embedding[:,0],
            embedding[:,1],
            c=colors,
            alpha=0.8)
        
        axes[i].set_xlabel("UMAP1")
        axes[i].set_ylabel("UMAP2")
        title = titles[i] if titles else f"UMAP Clustering {genes}"
        axes[i].set_title(title)
        axes[i].grid(True)

    # Hide unused subplots if any
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    
    return fig

## FUNCTION TO CREATE A DICTIONNARY THAT KEEPS TRACK OF THE ASSOCIATION READID, CLUSTER AND COLOR

def dict_id_cluster_color(
        df: pd.DataFrame,
        clusters : list,
        hex_colors: list,
):
    """
    Creating a dictionnary that keeps track of the association readid, cluster associated, and the color attributed
    df: dataframe that is binned, in a matrix form without any missing values (the dataframe used to cluster)
    clusters: the list containing the clusters after clustering(df)
    hex_colors: the hand defined color list (to not rely on the colormaps)
    """
    #creating a dictionnary to get the colors associated to the clusters

    unique_clusters = np.unique(clusters)
    cluster_colors = {
        cluster_id: hex_colors[i % len(hex_colors)]
        for i, cluster_id in enumerate(unique_clusters)
    }

#creating a dictionnary where the key is the readid and the values both the cluster id and the color
    read_dict = {
        read_id: {
            "cluster": cluster_id,
            "color": cluster_colors[cluster_id]
        }
        for read_id, cluster_id in zip(df.index.get_level_values('readid'), clusters)
    }
    return read_dict


## FUNCTION TO TURN A DICTIONNARY INTO A DATAFRAME

def dict_to_df(
        read_dict: dict
):
    """
    Transforming the dictionnary into a dataframe, (to be able to merge it with the original dataframe later on)
    read_dict: the dictionnary created with the function dict_id_cluster_color
    """
    df_dict = pd.DataFrame.from_dict(read_dict, orient='index')
    df_dict.index.name = 'readid'
    df_dict.reset_index(inplace=True)
    df_dict.rename(columns={"cluster": "locus_cluster", "color": "locus_cluster_color"}, inplace=True)
    
    return df_dict


## FUNCTION TO MERGE A DATAFRAME WITH THE DICTIONNARY-DATAFRAME
def merge(
        df,
        df_dict: pd.DataFrame  
):
    # Ensure readid is a column
    if "readid" not in df.columns:
        df = df.reset_index().rename(columns={"index": "readid"}).copy()
    else:
        df_merged = df.copy()    

    #Merging the two dataframes on'readid' to get the cluster and color information for each read
    df_merged = df_merged.merge(df_dict, on="readid", how="inner")
    df_merged.dropna(subset=['locus_cluster'], inplace=True)
    
    return df_merged


## FUNCTION TO PLOT THE AVERAGE METHYLATION PROFILE PER CLUSTER, USING THE COLORS ATTRIBUTED TO EACH CLUSTER

def plot_avg_methylation_profile(
    df: pd.DataFrame,
    df_dict: pd.DataFrame,  # dataframe made out of the read_dict
    start: int,
    end: int,
    center_coord: int,
    read_dict: dict # {read_id: {'cluster': cluster_id, 'color': hex_color}}
):
    """
    Plot average methylation profiles per cluster using cluster-specific colors.

    Parameters
    ----------
    df : pd.DataFrame
        Each row = read, columns = genomic positions (bins), values = mean methylation (0-1)
    start : int
        Start genomic coordinate for plotting
    end : int
        End genomic coordinate for plotting
    center_coord : int
        Central position (e.g., Pol2)
    partition : ig.VertexClustering
        Clustering result (used to determine number of clusters if needed)
    read_dict : dict
        Mapping of read IDs to cluster and color: {read_id: {'cluster': id, 'color': '#hex'}}
    """
    # -------------------------
    # Step 1: Prepare DataFrame
    # -------------------------
    df_avg= df.copy()
    df_avg = df_avg.reset_index(level='readid', drop=False) 
    df_avg.index = range(len(df_avg)) 

    # -------------------------
    # Step 2: Merge with df_map
    # -------------------------
    df_merged = pd.merge(df_avg, df_dict, on='readid', how='inner')

    # -------------------------
    # Step 3: Identify numeric position columns
    # -------------------------
    metadata_cols = ['readid', 'locus_cluster', 'locus_cluster_color']
    position_cols = [col for col in df_merged.columns if col not in metadata_cols]
    positions_sorted = sorted([int(col) for col in position_cols])

    # -------------------------
    # Step 4: Cluster colors mapping
    # -------------------------
    cluster_colors = {}
    for rid, info in read_dict.items():
        cid = int(info['cluster'])
        if cid not in cluster_colors:
            cluster_colors[cid] = info['color']

    # -------------------------
    # Step 5: Compute cluster proportions
    # -------------------------
    unique_clusters = sorted(df_merged['locus_cluster'].dropna().astype(int).unique())
    n_clusters = len(unique_clusters)
    proportion_df = (
        df_merged['locus_cluster']
        .value_counts(normalize=True)
        .mul(100)
        .reindex(unique_clusters, fill_value=0)
        .reset_index(name="percentage")
        .rename(columns={"index": "locus_cluster"})
    )

    # -------------------------
    # Step 6: Create subplots
    # -------------------------
    height_ratios = proportion_df['percentage'].values
    fig, axes = plt.subplots(n_clusters, 1, figsize=(10, 12), sharex=True,
                             gridspec_kw={'height_ratios': height_ratios})
    if n_clusters == 1:
        axes = [axes]

    # -------------------------
    # Step 7: Plot each cluster
    # -------------------------
    for i, cluster_id in enumerate(unique_clusters):
        ax = axes[i]
        cluster_rows = df_merged[df_merged['locus_cluster'] == cluster_id]

        if cluster_rows.empty:
            continue

        # Compute mean methylation only over position columns (integers!)
        df_meth = cluster_rows[positions_sorted]
        meth_sorted = df_meth.mean(axis=0).values

        # Smoothing
        meth_smooth = gaussian_filter1d(meth_sorted, sigma=0.5)

        # Cluster color
        color = cluster_rows['locus_cluster_color'].iloc[0]

        # Plot

        # Set background to black
        ax.set_facecolor('black')

        # # Fill under the curve with the cluster color
        ax.fill_between(positions_sorted, meth_smooth, color=color)

        # Optionally, overlay a line for contrast
        # ax.plot(positions_sorted, meth_smooth, color='white', linewidth=1)

        # Y-axis label with percentage
        percentage = proportion_df.loc[
            proportion_df['locus_cluster'] == cluster_id, 'percentage'
        ].values[0]
        ax.set_ylabel(f"Cluster {cluster_id} | {percentage:.1f}%", fontsize=8)

        # Axis limits and grid
        ax.set_ylim(0, 1.1)
        ax.set_xlim(min(positions_sorted), max(positions_sorted))
    
        # Assign new labels
        ax.set_xticklabels([str(start), "50", str(end)])
        ax.grid(True, axis='x', linestyle='--', linewidth=0.5, color='gray')

        # Reference lines
        ax.axhline(y=1, color='grey', linestyle='--', linewidth=0.5)
        ax.axvline(x=0, color="#C80028", linestyle="-", linewidth=2, label="Pol2 position")
        ax.xaxis.set_major_locator(MultipleLocator(100))


    # -------------------------
    # Step 8: Global labels & legend
    # -------------------------
    
    # Show x tick labels only on the bottom axis
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    # Ticks every 100 on the bottom axis
    axes[-1].xaxis.set_major_locator(MultipleLocator(100))

    # Format bottom tick labels as absolute genomic coordinates
    def rel_to_abs_label(x, pos):
        # Optionally use thousands separators: f"{int(x + center_coord):,}"
        return f"{int(x + center_coord)}"

    axes[-1].xaxis.set_major_formatter(FuncFormatter(rel_to_abs_label))

    # Hide any potential offset text
    axes[-1].get_xaxis().get_offset_text().set_visible(False)

    # Global labels & legend
    fig.text(0.95, 0.5, 'Methylation level', va='center', ha='center', rotation=-90, fontsize=10)
    fig.text(
        0.05, 0.5,
        "Cluster number | Cluster proportion\n(ordered by cluster mean methylation)",
        fontsize=10, multialignment='center', rotation=90, va='center', ha='center'
    )
    fig.text(0.5, 0.04, "Genomic coordinate", va='center', ha='center', fontsize=10)

    center_line = mlines.Line2D([], [], color='red', linestyle='-', linewidth=1, label='pol2 position')
    fig.legend(handles=[center_line], loc='upper right', bbox_to_anchor=(0.97, 0.955), fontsize=9, frameon=False)

    plt.subplots_adjust(hspace=0.05, top=0.92)
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    axes[-1].tick_params(axis='x', labelsize=8)
    fig.subplots_adjust(bottom=0.18)  # extra bottom margin to avoid clipping

    plt.show()


## FUNCTION TO CREATE TWO HEATMAPS THAT SUMMARIZE THE CLUSTERING RESULTS
def summary(
        df: pd.DataFrame,
        clusters: list,
):
    df_summary = df.copy()
    df_summary['locus_cluster'] = clusters

    total_reads = len(df_summary)  # <-- denominator for global %s

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    # --- First heatmap: per group (global percentages) ---
    summary_per_group = (
        df_summary[['locus_cluster']]
        .groupby(df_summary.index.get_level_values('group'))['locus_cluster']
        .value_counts()                           # raw counts
        .div(total_reads).mul(100)                # convert to % of ALL reads (not relative to the number of reads per group, but all the reads)
        .reset_index(name="percentage")
    )

    pivot_group = summary_per_group.pivot(
        index="group", columns="locus_cluster", values="percentage"
    ).fillna(0)

    sns.heatmap(pivot_group, annot=True, fmt=".2f", cmap="viridis", ax=axes[0])
    axes[0].set_ylabel("Group")
    axes[0].set_xlabel("Cluster")
    axes[0].set_title("Reads per cluster in each group (% of all reads)")

    # --- Second heatmap: per bulk cluster (global percentages) ---
    summary_per_bulk_cluster = (
        df_summary[['locus_cluster']]
        .groupby(df_summary.index.get_level_values('cluster'))['locus_cluster']
        .value_counts()
        .div(total_reads).mul(100)
        .reset_index(name="percentage")
    )

    pivot_cluster = summary_per_bulk_cluster.pivot(
        index="cluster", columns="locus_cluster", values="percentage"
    ).fillna(0)

    sns.heatmap(pivot_cluster, annot=True, fmt=".2f", cmap="viridis", ax=axes[1])
    axes[1].set_ylabel("Bulk Cluster")
    axes[1].set_xlabel("Cluster")
    axes[1].set_title("Reads per cluster in each bulk cluster (% of all reads)")

    plt.show()
