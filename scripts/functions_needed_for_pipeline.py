import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FuncFormatter
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.impute import KNNImputer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, pairwise_distances
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import umap
import igraph as ig
import leidenalg as la
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D 


#MANUALLY RECALCULATING POL2 POSITION AS THE MIDPOINT BETWEEN MOTIF_START AND MOTIF_END
def calculate_pol2_position(df):
    """Calculate the Pol II position as the midpoint between motif_start and motif_end."""
    if 'motif_start' in df.columns and 'motif_end' in df.columns:
        df['pol2_pos'] = ((pd.to_numeric(df['motif_start'], errors='coerce') +
                                 pd.to_numeric(df['motif_end'], errors='coerce')) / 2).round().astype('Int64')
    else:
        raise ValueError("DataFrame must contain 'motif_start' and 'motif_end' columns.")
    return df

## 1. PREPROCESS OF THE DATAFRAME FUNCTION

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
    else:
        print("'pol2' column not found; skipping conversion to binary marker.")

    # Create gene_tss column if columns exist
    if "gene" in df_proc.columns and "tss_pos" in df_proc.columns:
        df_proc["gene_tss"] = df_proc["gene"].astype(str) + "_" + df_proc["tss_pos"].astype(str)
        print('Gene_tss created from gene and tss_pos columns')
    else:
        # fallback if gene/tss_pos not present
        rid_col = 'readid' if 'readid' in df_proc.columns else None
        cstart_col = 'C_start' if 'C_start' in df_proc.columns else None
        if rid_col and cstart_col:
            df_proc['gene_tss'] = df_proc[rid_col].astype(str) + "_" + df_proc[cstart_col].astype(str)
            print('Gene_tss created from readid and C_start columns')
        else:
            # final fallback
            print('ERROR: Could not create gene_tss; required columns missing.')

    # Drop duplicates again after transformations
    df_proc = df_proc.drop_duplicates()

    print(f"Final shape: {df_proc.shape}")
    print(f"NUmber of gene_tss: {df_proc['gene_tss'].nunique() if 'gene_tss' in df_proc.columns else 'N/A'}")

    return df_proc


## 2. FUNCTION TO FILTER ONLY THE GENES THAT HAVE SUFFICIENT READS THAT OVERLAP ON THE SAME GENOMIC REGION, CENTERED ON THE MIDDLE
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


## 3. BINNING THE DATAFRAME AND TURNING IT INTO MATRIX FORM
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


## 4. PROCESSING -- MARKING THE METHYLATION OF READS BY BINNED REGIONS -- GIVING OUT A DATAFRAME

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
        if 'cluster' not in df.columns:
            index_list = ['gene_tss','readid','group']
        else:
            index_list = ['gene_tss','readid','group','cluster']
        df = bin_then_matrix(df, indexes= index_list, bin_size= bin_size)

     
    print(f"Shape of the dataframe : {df.shape}")
    
    return df


## 5. TO HANDLE THE NAN VALUES ONCE IN MATRIX FORM

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


##6. FUNCTION TO PREPARE THE DATAFRAME IN LONG FORMAT FOR PLOTTING READS

def preprocess_long_for_plot(
    df: pd.DataFrame,
    include_locus_cluster: bool = False,
    # --- Filtering options ---
    filter_outliers: bool = False,
    max_span_bp: float | None = None,   # e.g., 4000; if None, use span_quantile
    span_quantile: float = 0.99,        # used if max_span_bp is None
    require_center_inside: bool = True, # enforce read_start <= pol2_pos <= read_end
    min_cpg: int = 1,                   # minimum CpGs per read
    cpg_window_bp: float | None = None, # constrain CpGs to ±window around center
    return_stats: bool = False          # optionally return per-read stats
):
    """
    Prepare a long-format DataFrame for plotting reads (one row per CpG per read),
    and optionally drop problematic reads (extreme spans, miscentered, or too few CpGs).

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with at least these columns:
        ['gene_tss', 'group', 'readid', 'cluster', 'C_start', 'meth', 'pol2_pos'].
    include_locus_cluster : bool, default=False
        If True and 'locus_cluster' is present, it will be retained.
    filter_outliers : bool, default=False
        Whether to apply filtering of problematic reads based on read span, centering, etc.
    max_span_bp : float or None, default=None
        Maximum allowed span of reads in base pairs. If None, uses span_quantile.
    span_quantile : float, default=0.99
        Quantile used to determine span threshold if max_span_bp is None.
    require_center_inside : bool, default=True
        Whether to enforce that each read covers the center position (0).
    min_cpg : int, default=1
        Minimum number of CpG sites required per read.
    cpg_window_bp : float or None, default=None
        If provided, constrains CpGs to lie within ±window around center.
    return_stats : bool, default=False
        If True, return per-read stats alongside the filtered DataFrame.

    Returns
    -------
    df_clean : pandas.DataFrame
        Processed DataFrame ready for plotting.
    stats : pandas.DataFrame, optional
        Per-read statistics if return_stats=True.
    """

    # --- Initial cleanup ---
    df = df.reset_index(drop=True).copy()

    # --- Keep relevant columns ---
    base_cols = ["gene_tss", "group", "readid", "cluster", "C_start", "meth", "pol2_pos"]
    keep_cols = [c for c in df.columns if c in base_cols]

    if include_locus_cluster and "locus_cluster" in df.columns:
        keep_cols.append("locus_cluster")

    df = df[keep_cols]

    # --- Ensure numeric types ---
    for c in ["C_start", "pol2_pos"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- Drop rows missing essentials ---
    df = df.dropna(subset=["readid", "C_start", "pol2_pos"])

    # --- Compute per-read min/max span (from CpG coordinates) ---
    span = (
        df.groupby("readid")["C_start"]
        .agg(["min", "max"])
        .rename(columns={"min": "read_start", "max": "read_end"})
    )
    df = df.merge(span, on="readid", how="left")

    # --- Compute per-read center (first pol2_pos per read) ---
    df["center"] = df.groupby("readid")["pol2_pos"].transform("first")

    # --- Shifted coordinates (relative to center) ---
    df["C_start_shifted"] = df["C_start"] - df["center"]
    df["read_start_shifted"] = df["read_start"] - df["center"]
    df["read_end_shifted"] = df["read_end"] - df["center"]

    # --- Optional filtering ---
    stats = None
    if filter_outliers:
        # Per-read summary stats
        stats = df.groupby("readid").agg(
            start=("read_start_shifted", "first"),
            end=("read_end_shifted", "first"),
            n_cpg=("C_start_shifted", "count"),
            min_c=("C_start_shifted", "min"),
            max_c=("C_start_shifted", "max"),
        )
        stats["span"] = stats["end"] - stats["start"]

        # Determine span threshold
        if max_span_bp is None:
            span_thr = stats["span"].quantile(span_quantile)
        else:
            span_thr = float(max_span_bp)

        # Base mask
        mask = (
            (stats["n_cpg"] >= int(min_cpg)) &
            (stats["span"] > 0) &
            (stats["span"] <= span_thr)
        )

        # Require center to be inside
        if require_center_inside:
            mask &= (stats["start"] <= 0) & (stats["end"] >= 0)

        # Constrain CpG window
        if cpg_window_bp is not None:
            w = float(cpg_window_bp)
            mask &= (stats["min_c"] >= -w) & (stats["max_c"] <= w)

        # Keep only valid reads
        keep_reads = stats.index[mask]
        df = df[df["readid"].isin(keep_reads)].copy()

        # Return filtered stats if requested
        if return_stats:
            stats = stats.loc[keep_reads]

    # --- Return ---
    if return_stats:
        return df, stats
    else:
        return df


## 7. FUNCTION TO PLOT THE READS IN LONG FORMAT    

def plot_reads_long(
    df,
    filters=None,
    facet_by=None,
    color_by=None,
    hex_colors=None,
    gene='Gene',
    max_facets=14,
    figsize=None  # otherwise a tuple, for ex (30,20)
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

    if figsize is None:
        fig = plt.figure(figsize=(30, total_height))
    else:
        fig = plt.figure(figsize=figsize)

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
                xmin=sub["read_start_shifted"].iloc[0]-100,
                xmax=sub["read_end_shifted"].iloc[0]+100,
                color=span_color,
                linewidth=1.6
            )

            # CpG dots
            for _, row in sub.iterrows():
                dot_color = "white" if row["meth"] == 1 else "black"
                ax.plot(row["C_start_shifted"], idx, "o", color=dot_color, markersize=3)

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
        ax.set_title(f" {gene} : Read-level methylation centered on Pol2 ({title})", fontsize=16)
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

    # return fig


## 8. SMALL FUNCTION TO GET THE LIST OF ALL THE GENES NAMES OF A DATAFRAME AND THE LIST OF ALL THE SUBDATAFRAME OF THESE GENES

def get_genes_list(df):
    gene_list_names= [] #the list that will take in the genes names
    gene_list_df=[] #the list that will take in the dataframes associated to the genes

    for gene, sub_df in df.groupby(df.index.get_level_values("gene_tss")):
        gene_list_names.append(gene)
        gene_list_df.append(sub_df)

    return gene_list_names,gene_list_df

def get_groups_df_list(df):
    group_list_names= [] #the list that will take in the groups names
    group_list_df=[] #the list that will take in the dataframes associated to the groups

    for group,sub_df in df.groupby(df.index.get_level_values("group")):
        group_list_names.append(group)
        group_list_df.append(sub_df)

    return group_list_names, group_list_df

## function called afterwards in clustering_final
def compute_cluster_metrics(
        X_pca: np.ndarray,
        clusters: np.ndarray,
        metric: str,
        graph,            # igraph.Graph (g)
        partition         # leidenalg partition (part)
    ) -> dict:
        """
        Safe computation of clustering quality metrics.
        - Returns None for silhouette/CH/DB when invalid (e.g., single cluster).
        - Falls back to precomputed distances for silhouette if needed.
        """
        metrics = {}
        labels = np.asarray(clusters)
        n_labels = int(np.unique(labels).size)
        n_samples = int(X_pca.shape[0])

        # Silhouette: requires at least 2 clusters and n_samples >= 2
        if n_labels >= 2 and n_samples >= 2:
            try:
                metrics['silhouette'] = float(silhouette_score(X_pca, labels, metric=metric))
            except Exception:
                try:
                    D = pairwise_distances(X_pca, metric=metric)
                    metrics['silhouette'] = float(silhouette_score(D, labels, metric='precomputed'))
                except Exception:
                    metrics['silhouette'] = None
        else:
            metrics['silhouette'] = None

        # Calinski–Harabasz: requires at least 2 clusters
        if n_labels >= 2 and n_samples >= 2:
            try:
                metrics['calinski_harabasz'] = float(calinski_harabasz_score(X_pca, labels))
            except Exception:
                metrics['calinski_harabasz'] = None
        else:
            metrics['calinski_harabasz'] = None

        # Davies–Bouldin: requires at least 2 clusters
        if n_labels >= 2 and n_samples >= 2:
            try:
                metrics['davies_bouldin'] = float(davies_bouldin_score(X_pca, labels))
            except Exception:
                metrics['davies_bouldin'] = None
        else:
            metrics['davies_bouldin'] = None

        # Leiden objective value
        try:
            metrics['leiden_quality'] = float(partition.quality())
        except Exception:
            metrics['leiden_quality'] = None

        # Weighted modularity
        try:
            metrics['modularity'] = graph.modularity(labels.tolist(), weights=graph.es['weight'])
        except Exception:
            metrics['modularity'] = None

        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        metrics['cluster_sizes'] = dict(zip(unique.tolist(), counts.tolist()))

        return metrics

# ------------------------------
# Build positional weights
# ------------------------------
def build_positional_weights(
    columns,                      # iterable of bin positions (e.g., DataFrame.columns)
    window_bp: int = 500,         # half-window around center to emphasize (e.g., +/- 500 bp)
    center: int = 0,              # Pol2 position (0 if already centered)
    mode: str = "gaussian",       # 'gaussian' or 'box'
    inside_weight: float = 1.0,   # weight inside window (for 'box')
    outside_weight: float = 0.2,  # weight outside window (for 'box')
    sigma_bp: float | None = None,# Gaussian sigma in bp (defaults to window_bp/2)
    normalize_mean: bool = True        # normalize to mean ~ 1.0
) -> pd.Series:
    """
    Create a positional weight vector indexed by columns (positions), emphasizing +/- window_bp around center.
    Returns a pd.Series aligned to the provided columns.
    """
    pos = pd.to_numeric(pd.Series(list(columns), dtype=str).str.strip(), errors="coerce").astype(int)
    if mode == "gaussian":
        # Gaussian weighting: exp(- (pos - center)^2 / (2 * sigma^2))
        sigma = (window_bp / 2.0) if sigma_bp is None else float(sigma_bp)
        w = np.exp(-((pos - center) ** 2) / (2.0 * sigma ** 2))
        w = outside_weight + (inside_weight - outside_weight) * w
    elif mode == "box":
        w = np.where(np.abs(pos - center) <= window_bp, inside_weight, outside_weight)
    else:
        raise ValueError("mode must be 'gaussian' or 'box'")

    if normalize_mean:
        mean_val = np.mean(w) if np.mean(w) > 0 else 1.0
        w = w / mean_val
    return pd.Series(w, index=pos)


## 3. IMPROVED CLUSTERING ALGORITHM, FOLLOWING THE STEPS: PCA --> ADAPTIVE KNN GRAPH (WEIGHTED) --> LEIDEN
def clustering_final(
    df,
    n_neighbors=15,
    nan_threshold : float = 0.7, #to drop the bins that have too many nan
    nan_method : str = 'drop', #by default dropping the rows that contain nan, but can use 'impute' too
    scaling : bool = False, #whether to scale the data or not
    pca_or_not = True, #whether to do a pca or not
    n_pcs= None ,# if None: by default 0.95 variance, otherwise int number of pcs to keep if pca is done
    metric='cosine', #cosine or euclidean
    transform='none', # 'none', 'logit', or 'arcsine'
    kernel_type='laplacian', # 'laplacian' or 'gaussian'
    leiden_resolution=1.0,
    seed=42,
    pos_weights: pd.Series | None = None,     # built via build_positional_weights(df.columns, ...)
    weight_stage: str = 'pre_pca'             # 'pre_pca' (default) or 'post_pca' (rare)
):
    """
    Perform Leiden clustering on PCA-reduced data with adaptive similarity weighting.
    And with optional positional weights (column-wise),
    which emphasize bins around Pol2 by scaling features by sqrt(weight).

    pos_weights:
        pd.Series indexed by df.columns (positions). If provided, applied as:
        X *= sqrt(weight) per column at the specified stage (pre_pca recommended).

    Parameters
    ----------
    df : pandas.DataFrame
        Input data (numeric).
    n_neighbors : int, default=15
        Number of neighbors for kNN graph construction.
    n_pcs : int, default=30
        Number of principal components to retain.
    metric : str, default='cosine'
        Distance metric for nearest neighbors.
    transform : {'none', 'logit', 'sqrt'}, default='none'
        Optional transformation applied to data before PCA.
    leiden_resolution : float, default=1.0
        Resolution parameter for the Leiden algorithm.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    clusters : np.ndarray
        Cluster assignments for each observation.
    part : leidenalg.Partition
        Leiden partition object.
    X_pca : np.ndarray
        PCA-transformed coordinates.
    """
    
    # 0) Handle NaNs via your function
    df_proc = handling_NaN(df, nan_threshold, nan_method)

    # 1) X matrix
    X = df_proc.to_numpy(dtype=float)
    if np.isnan(X).any():
        raise ValueError("NaNs remain after handling_NaN; impute explicitly before PCA.")

    # 2) Transform
    if transform == 'logit':
        Xc = np.clip(X, 1e-3, 1 - 1e-3)
        Xc = np.log(Xc / (1 - Xc))
    elif transform == 'arcsine':
        Xc = np.arcsin(np.sqrt(np.clip(X, 0.0, 1.0)))
    elif transform == 'none':
        Xc = X
    else:
        raise ValueError("transform must be 'none', 'logit', or 'arcsine'")

    # 2b) Optional scaling (column-wise z-score) — usually keep False if using positional weights
    if scaling:
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xc = scaler.fit_transform(Xc)

    # 2c) Apply positional weights (pre-PCA): Xc *= sqrt(weights) per column
    if pos_weights is not None and weight_stage == 'pre_pca':
        w_ser = pd.Series(pos_weights, index=df_proc.columns).reindex(df_proc.columns).astype(float).fillna(1.0)
        sw = np.sqrt(np.maximum(w_ser.to_numpy(), 0.0))
        Xc = Xc * sw[None, :]

    # 3) PCA
    if pca_or_not:
        if n_pcs is None:
            pca = PCA(n_components=0.95, svd_solver='auto', random_state=seed)
        elif isinstance(n_pcs, int) and n_pcs > 0:
            n_pcs = min(n_pcs, min(Xc.shape) - 1)
            pca = PCA(n_components=n_pcs, svd_solver='auto', random_state=seed)
        else:
            raise ValueError("n_pcs must be None or positive int.")
        X_pca = pca.fit_transform(np.nan_to_num(Xc, nan=0.0))
        print("Number of components chosen:", pca.n_components_)
    else:
        X_pca = Xc  # no PCA

    # 3b) Optional post-PCA weighting (rare; only if you specifically want to warp the embedding)
    if pos_weights is not None and weight_stage == 'post_pca':
        # Only makes sense if PCA wasn't used (feature space must align to original bins).
        # Generally discouraged; prefer 'pre_pca'.
        w_ser = pd.Series(pos_weights, index=df_proc.columns).reindex(df_proc.columns).astype(float).fillna(1.0)
        sw = np.sqrt(np.maximum(w_ser.to_numpy(), 0.0))
        # If no PCA, X_pca has same feature dimension as original columns.
        if not pca_or_not and X_pca.shape[1] == sw.size:
            X_pca = X_pca * sw[None, :]

    # 3c) For cosine without PCA, L2-normalize rows (optional; cosine is scale-invariant)
    if metric == 'cosine' and not pca_or_not:
        X_pca = normalize(X_pca, norm='l2', axis=1)

    N = X_pca.shape[0]


    # 4) kNN graph
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nn.fit(X_pca)
    dist, idx = nn.kneighbors(X_pca, return_distance=True)

    # 5) Kernels
    self_included = np.all(idx[:, 0] == np.arange(N))
    eps = 1e-12
    if kernel_type == 'laplacian':
        dist_for_scale = dist[:, 1:] if self_included else dist
        tau = np.median(dist_for_scale, axis=1) + 1e-6
        sim = np.exp(-dist / tau[:, None])
    elif kernel_type == 'gaussian':
        sigma = dist[:, -1] + eps
        sigma_j = sigma[idx]
        sigma_i = sigma[:, None]
        sim = np.exp(- (dist ** 2) / (sigma_i * sigma_j + eps))
    else:
        raise ValueError("kernel_type must be 'laplacian' or 'gaussian'")


    # 6) Undirected weighted graph (union kNN with max weight)
    edges = {}
    for i in range(N):
        if self_included:
            neigh_idx = idx[i, 1:]; neigh_w = sim[i, 1:]
        else:
            neigh_idx = idx[i, :];  neigh_w = sim[i, :]

        for j, w in zip(neigh_idx, neigh_w):
            if i == j or w <= 0:
                continue
            a, b = (i, j) if i < j else (j, i)
            edges[(a, b)] = max(edges.get((a, b), 0.0), float(w))

    e_list = list(edges.keys())
    w_list = [edges[e] for e in e_list]

    g = ig.Graph(n=N, edges=e_list, directed=False)
    g.es["weight"] = w_list

    # 7) Leiden
    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights=g.es['weight'],
        resolution_parameter=leiden_resolution,
        seed=seed
    )
    clusters = np.array(part.membership)

    # 8) Metrics
    metrics = compute_cluster_metrics(
        X_pca=X_pca,
        clusters=clusters,
        metric=metric,
        graph=g,
        partition=part
    )

    return df_proc, X_pca, part, clusters, metrics


## PLOTTING THE UMAP CLUSTERING

def plot_umap(
    X,
    clusters,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',      # for Pipeline 2, use 'euclidean'
    transform=None,          # None | 'logit' | 'arcsine' (use only if X are raw proportions)
    n_pcs=None,              # set if X are raw features; None if X are already PCs
    standardize=False,       # True if using raw features without PCA
    seed=42,
    palette=None,            # optional list/array or matplotlib colormap name
    title="UMAP embedding",
    gene = 'Gene'
):

    X_in = np.asarray(X, dtype=float)

    # Optional transform for proportions (only if X are raw fractions)
    if transform is not None:
        if transform == 'logit':
            Xc = np.clip(X_in, 1e-3, 1 - 1e-3)
            X_in = np.log(Xc / (1 - Xc))
        elif transform == 'arcsine':
            X_in = np.arcsin(np.sqrt(np.clip(X_in, 0.0, 1.0)))
        else:
            raise ValueError("transform must be None, 'logit', or 'arcsine'")

    # Optional standardization (useful if running UMAP on raw features)
    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_in = scaler.fit_transform(X_in)

    # Optional PCA (skip if X are already PCA scores)
    if n_pcs is not None and 0 < n_pcs < X_in.shape[1]:
        pca = PCA(n_components=n_pcs, random_state=seed)
        X_umap = pca.fit_transform(X_in)
    else:
        X_umap = X_in

    # Sanity checks for metric
    if metric == 'jaccard':
        # Warn if data are not binary
        if not np.array_equal(X_umap, X_umap.astype(bool)) and not np.array_equal(
            X_umap, (X_umap > 0).astype(int)
        ):
            raise ValueError(
                "Jaccard metric requires binary data; "
                "use euclidean/cosine/correlation for continuous features."
            )

    # UMAP embedding
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed
    )
    embedding = reducer.fit_transform(X_umap)

    # Colors
    unique_clusters = np.unique(clusters)
    if palette is None:
        # fallback to a matplotlib qualitative colormap
        cmap = plt.get_cmap('tab20')
        color_map = {c: cmap(i % cmap.N) for i, c in enumerate(unique_clusters)}
    elif isinstance(palette, str):
        cmap = plt.get_cmap(palette)
        color_map = {c: cmap(i % cmap.N) for i, c in enumerate(unique_clusters)}
    else:
        # palette is a list/array
        color_map = {c: palette[i % len(palette)] for i, c in enumerate(unique_clusters)}

    colors = [color_map[c] for c in clusters]

    # Plot
    fig = plt.figure(figsize=(10, 7))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        alpha=0.85,
    )
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title(f'{gene} : {title}')
    plt.grid()

    # Legend
    handles = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=str(c),
            markerfacecolor=color_map[c],
            markersize=8
        )
        for c in unique_clusters
    ]
    plt.legend(
        handles=handles,
        title="Cluster",
        bbox_to_anchor=(1.02, 1.0),
        loc='upper left',
        frameon=False
    )
    plt.tight_layout()

    return embedding, fig

## 1. DICTIONNARY TO KEEP TRACK OF THE ASSOCIATION READID, CLUSTER AND COLOR
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

## 2. TRANSFORMING THE DICTIONNARY INTO A DATAFRAME
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

## 3. MERGING THE DICTIONNARY DATAFRAME WITH THE ORIGINAL DATAFRAME (TO GET THE CLUSTER AND COLOR INFORMATION FOR EACH READ)
def merge(
        df,
        df_dict: pd.DataFrame  
):
    # Ensure readid is a column
    if "readid" not in df.columns:
        df_merged = df.reset_index().rename(columns={"index": "readid"}).copy()
    else:
        df_merged = df.copy()    

    #Merging the two dataframes on'readid' to get the cluster and color information for each read
    df_merged = df_merged.merge(df_dict, on="readid", how="inner")
    df_merged.dropna(subset=['locus_cluster'], inplace=True)
    
    return df_merged

## 4. FUNCTION TO GET THE START, END AND CENTER COORDINATES OF A GENE
def start_end_center(
        df: pd.DataFrame
):
    if 'read_start' in df.columns and 'read_end' in df.columns and 'pol2_pos' in df.columns:
        start = df['read_start'].min()
        end = df['read_end'].max()
        center_coord = df['pol2_pos'].unique()[0]  

        return start, end, center_coord
    else:
        raise ValueError("DataFrame must contain 'read_start', 'read_end', and 'pol2_pos' columns.")


## 1. TO COMPARE THE LOCUS SPECIFIC CLUSTERS RELATIVELY TO THE BULK CLUSTERS AND THE GROUPS

def summary(
        df: pd.DataFrame,
        clusters: list,
        gene='Gene'
):
    df_summary = df.copy()
    df_summary['locus_cluster'] = clusters

    total_reads = len(df_summary)  # <-- denominator for global %s

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    # --- First heatmap: per group (global percentages) ---
    if 'group' in df_summary.index.names:
        # group is part of the index
        summary_per_group = (
            df_summary[['locus_cluster']]
            .groupby(df_summary.index.get_level_values('group'))['locus_cluster']
            .value_counts()
            .div(total_reads).mul(100)
            .reset_index(name="percentage")
        )
    else:
        # group is a normal column
        summary_per_group = (
            df_summary[['group', 'locus_cluster']]
            .groupby('group')['locus_cluster']
            .value_counts()
            .div(total_reads).mul(100)
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

    if 'cluster' in df_summary.index.names:
        # group is part of the index
        summary_per_bulk_cluster = (
            df_summary[['locus_cluster']]
            .groupby(df_summary.index.get_level_values('cluster'))['locus_cluster']
            .value_counts()
            .div(total_reads).mul(100)
            .reset_index(name="percentage")
        )
    else:
        # group is a normal column
        summary_per_bulk_cluster = (
            df_summary[['cluster', 'locus_cluster']]
            .groupby('cluster')['locus_cluster']
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

    plt.title(f'{gene} : Heatmaps of repartitions')
    plt.show()


## IN ORDER TO GET THE AVERAGE METHYLATION ON BULK DATA 
def compute_bulk_centroids(
    df: pd.DataFrame,
    use_cov_for_proportion: bool = True
):
    """
    Compute per-cluster centroids across the entire bulk table.

    Input df columns: ['cluster','C_pos','meth','cov'].
    Returns:
      P_df: clusters × bins matrix (mean methylation per bin)
      cov_df: clusters × bins matrix (coverage per bin)
      meta_df: per-cluster summary with ['cluster_id','n_rows_or_cov','proportion']
      positions: sorted integer bin coordinates (C_pos)
    """
    required = {'cluster','C_pos','meth','cov'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataframe missing columns: {missing}")

    d = df.copy()
    d['C_pos'] = pd.to_numeric(d['C_pos'], errors='coerce')
    d['meth']  = pd.to_numeric(d['meth'],  errors='coerce')
    d['cov']   = pd.to_numeric(d['cov'],   errors='coerce')
    d = d.dropna(subset=['cluster','C_pos','meth'])

    # Aggregate duplicates at the same position within cluster
    agg = (
        d.groupby(['cluster','C_pos'], as_index=False)
         .agg(meth_mean=('meth','mean'),
              cov_agg=('cov','first'))  # use 'sum' if cov varies per row
    )

    # Pivot to wide: cluster × position
    P_df = agg.pivot(index='cluster', columns='C_pos', values='meth_mean')
    cov_df = agg.pivot(index='cluster', columns='C_pos', values='cov_agg')

    # Sort columns and convert to ints
    positions = np.array(sorted(P_df.columns.astype(int)), dtype=int)
    P_df = P_df.reindex(columns=positions)
    cov_df = cov_df.reindex(columns=positions)

    # Proportions (height ratios)
    g = d.groupby('cluster')
    if use_cov_for_proportion:
        cov_first = g['cov'].first()
        cov_var   = g['cov'].var()
        cov_tot   = g['cov'].sum() if np.nanmax(cov_var.fillna(0)) > 0 else cov_first
        total_cov = float(cov_tot.sum()) if cov_tot.sum() is not None else 0.0
        proportions = (cov_tot / (total_cov if total_cov > 0 else 1.0)) * 100.0
        n_metric = cov_tot.astype(float)
        n_label = 'cov_total'
    else:
        counts = g.size()
        proportions = (counts / counts.sum()) * 100.0
        n_metric = counts.astype(float)
        n_label = 'n_rows'

    meta_rows = []
    for cid in P_df.index:
        meta_rows.append({
            'cluster_id': cid if isinstance(cid, (int, np.integer)) else str(cid),
            n_label: float(n_metric.get(cid, np.nan)),
            'proportion': float(proportions.get(cid, 0.0))
        })
    meta_df = pd.DataFrame(meta_rows).sort_values('cluster_id').reset_index(drop=True)

    return P_df, cov_df, meta_df, positions

## IN ORDER TO GET AVERAGE METHYLATION VALUE FOR SINGLE GENES

def compute_gene_centroids(
    df_reads: pd.DataFrame,
    df_map: pd.DataFrame,  # ['readid', 'locus_cluster', 'locus_cluster_color']
    gene_tss : str= None ,
    group: str= None ,
    read_id_col: str = 'readid',
    cluster_col: str = 'locus_cluster',
    color_col: str = 'locus_cluster_color',
    extra_meta_cols: list | None = None  # e.g., ['gene_tss', 'group']
):
    """
    Compute per-cluster average methylation per bin (and coverage) for one gene.

    Returns
    -------
    profiles_df : pd.DataFrame
        Mean methylation per cluster × bin.
    coverage_df : pd.DataFrame
        Number of reads with non-NaN methylation per cluster × bin.
    meta_df : pd.DataFrame
        Metadata for each cluster: [gene_tss, cluster_id, n_reads, proportion, color].
    positions : np.ndarray
        Sorted numeric bin positions (integers).
    """
    # --- Ensure read IDs are present as a column ---
    if read_id_col not in df_reads.columns:
        df_w = df_reads.reset_index().rename(columns={'index': read_id_col})
    else:
        df_w = df_reads.copy()

    # --- Merge cluster/color mapping ---
    dfg = pd.merge(
        df_w,
        df_map[[read_id_col, cluster_col, color_col]],
        on=read_id_col,
        how='inner'
    )

    if dfg.empty:
        raise ValueError("No overlapping reads between df_reads and df_map.")

    # --- Identify metadata columns ---
    meta_cols = {read_id_col, cluster_col, color_col}
    if extra_meta_cols:
        meta_cols |= set(extra_meta_cols)

    # --- Select candidate bin columns ---
    candidate_cols = [c for c in dfg.columns if c not in meta_cols]

    # --- Keep only columns with numeric names AND numeric data ---
    def int_like(name):
        return (
            isinstance(name, (int, np.integer))
            or (isinstance(name, str) and name.strip().lstrip('-').isdigit())
        )

    bin_cols_num = []
    positions = []
    for c in candidate_cols:
        if int_like(c) and pd.api.types.is_numeric_dtype(dfg[c]):
            bin_cols_num.append(c)
            positions.append(int(c) if not isinstance(c, (int, np.integer)) else int(c))

    if not bin_cols_num:
        raise ValueError(
            "No numeric bin columns found. Check df_reads columns and extra_meta_cols."
        )

    positions = np.array(sorted(positions), dtype=int)

    # --- Build clean numeric-column version ---
    col_map = {
        c: (int(c) if not isinstance(c, (int, np.integer)) else int(c))
        for c in bin_cols_num
    }
    dfg_bins = dfg[[cluster_col, color_col] + bin_cols_num].rename(columns=col_map)
    dfg_bins = dfg_bins[[cluster_col, color_col] + positions.tolist()]

    # --- Group by cluster and compute stats ---
    profiles = {}
    coverage = {}
    colors = {}
    counts_per_cluster = {}

    for cid, sub in dfg_bins.groupby(cluster_col):
        vals = sub[positions]
        profiles[cid] = vals.mean(axis=0, skipna=True)
        coverage[cid] = vals.notna().sum(axis=0)
        colors[cid] = (
            sub[color_col].iloc[0]
            if color_col in sub.columns and not sub[color_col].isna().all()
            else None
        )
        counts_per_cluster[cid] = int(len(sub))

    profiles_df = pd.DataFrame(profiles).T.reindex(columns=positions)
    coverage_df = pd.DataFrame(coverage).T.reindex(columns=positions)

    profiles_df.index.name = 'cluster'
    profiles_df.columns.name = 'C_pos'

    # --- Build meta table ---
    total_reads = sum(counts_per_cluster.values())
    meta_rows = []
    for cid in profiles_df.index:
        n_reads = counts_per_cluster.get(cid, 0)
        pct = (n_reads / total_reads * 100.0) if total_reads > 0 else 0.0

        if gene_tss is not None :
            meta_rows.append({
                'gene_tss': gene_tss,
                'cluster_id': int(cid),
                'n_reads': int(n_reads),
                'proportion': float(pct),
                'cluster_color': colors.get(cid)
            })
        elif group is not None :
            meta_rows.append({
                'group': group,
                'cluster_id': int(cid),
                'n_reads': int(n_reads),
                'proportion': float(pct),
                'cluster_color': colors.get(cid)
            })
        else:
            meta_rows.append({
                'cluster_id': int(cid),
                'n_reads': int(n_reads),
                'proportion': float(pct),
                'cluster_color': colors.get(cid)
            })

    meta_df = (
        pd.DataFrame(meta_rows)
        .sort_values('cluster_id')
        .reset_index(drop=True)
    )


    return profiles_df, coverage_df, meta_df, positions

## TO PLOT THE AVERAGE METHYLATION PATTERN

def plot_centroids_with_shading(
    P_df: pd.DataFrame,                 # index = cluster_id, columns = positions (int), values = mean methylation
    positions: np.ndarray,              # sorted integer positions
    meta_df: pd.DataFrame,              # must include ['cluster_id', 'proportion']; optional ['n_reads', 'cluster_color']
    coverage_df: pd.DataFrame | None = None,  # same shape as P_df; per-bin contributing read counts
    hex_colors=None,
    proportional_height: bool = True,  # if True, cluster panel heights proportional to 'proportion' in meta_df
    smooth_sigma: float = 0.5,
    start: int | None = None,
    end: int | None = None,
    title: str = "Average DNA Methylation Profiles per Cluster",
    show_pol2_line: bool = True,
    shade_missing: bool = True,          # enable/disable shading
    missingness_threshold: float = 0.7,  # shade sites where missingness >= threshold
    bin_width: int | None = None         # if None, inferred from median bin spacing
):
    # --- Clusters and proportions ---
    clusters = list(P_df.index)
    n_clusters = len(clusters)

    if 'proportion' in meta_df.columns:
        prop_map = {int(row['cluster_id']): float(row['proportion'])
                    for _, row in meta_df.iterrows()}
    else:
        prop_map = {int(cid): 100.0 / max(n_clusters, 1) for cid in clusters}

    # --- Colors ---
    if 'cluster_color' in meta_df.columns and meta_df['cluster_color'].notna().any():
        color_map = {int(row['cluster_id']): row['cluster_color']
                     for _, row in meta_df.iterrows()}
    else:
        import matplotlib.cm as cm
        if hex_colors is None:
            cmap = cm.get_cmap('tab20', n_clusters)
            hex_colors = [cm.colors.to_hex(cmap(i)) for i in range(n_clusters)]

        color_map = {cid: hex_colors[i % len(hex_colors)]
                     for i, cid in enumerate(clusters)}
        
    # Ensure meta_df has a 'cluster_color' column and fill (preserve existing, fill missing)
    if 'cluster_color' not in meta_df.columns:
        meta_df['cluster_color'] = np.nan

    meta_df['cluster_color'] = meta_df['cluster_color'].fillna(meta_df['cluster_id'].map(color_map))

    # --- Height ratios from proportions ---
    height_ratios = [max(1e-3, float(prop_map.get(int(cid), 0.0)))
                     for cid in clusters]

    # --- Infer bin width ---
    if bin_width is None and len(positions) > 1:
        diffs = np.diff(positions)
        bin_width = int(np.median(diffs)) if len(diffs) else 1
    if bin_width is None:
        bin_width = 1

    # --- Estimate per-cluster size for missingness ---
    n_reads_map = {}
    if 'n_reads' in meta_df.columns:
        n_reads_map = {int(row['cluster_id']): int(row['n_reads'])
                       for _, row in meta_df.iterrows()}
    elif coverage_df is not None:
        for cid in clusters:
            try:
                n_reads_map[int(cid)] = int(np.nanmax(
                    coverage_df.loc[cid].to_numpy(float)))
            except Exception:
                n_reads_map[int(cid)] = 0

    # --- Create figure --- if i want the cluster profiles height to be proportional to the number of reads
    if proportional_height:
        fig, axes = plt.subplots(
            n_clusters, 1, 
            figsize=(8, 10), 
            sharex=True,
            gridspec_kw={'height_ratios': height_ratios}
        )

    elif not proportional_height:
    # create figure for everything is of the same height:
        panel_height_in = 2.0  # height per cluster panel (inches), adjust as needed

        fig, axes = plt.subplots(
            n_clusters, 1,
            figsize=(16, panel_height_in * max(1, n_clusters)),  # total height scales with number of clusters
            sharex=True,
            gridspec_kw={'height_ratios': [1] * n_clusters}      # equal height for all panels
        )

    if n_clusters == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, y=0.97)

    # --- Plot each cluster ---
    for i, cid in enumerate(clusters):
        ax = axes[i]
        y = P_df.loc[cid].to_numpy(float)
        x = positions
        y_smooth = (gaussian_filter1d(y, sigma=float(smooth_sigma))
                    if smooth_sigma and smooth_sigma > 0 else y)

        # --- Shading of high missingness sites ---
        if shade_missing and coverage_df is not None:
            cs = int(n_reads_map.get(int(cid), 0))
            if cs > 0:
                cov = coverage_df.loc[cid].to_numpy(float)
                miss_frac = 1.0 - (cov / cs)
                mask = miss_frac >= float(missingness_threshold)

                runs = []
                run_start = None
                prev = None
                for p, ok in zip(x, mask):
                    if ok and run_start is None:
                        run_start = p
                        prev = p
                    elif ok:
                        prev = p
                    elif (not ok) and run_start is not None:
                        runs.append((run_start, prev))
                        run_start = None
                if run_start is not None:
                    runs.append((run_start, prev))

                for a, b in runs:
                    ax.axvspan(a - bin_width / 2, b + bin_width / 2,
                               color="#969696", alpha=0.7, zorder=0.9)

        # --- Plot profile ---
        ax.set_facecolor(color_map.get(cid, '#cccccc'))
        ax.fill_between(x, y_smooth, color='white')

        # --- Axes, labels ---
        pct = float(prop_map.get(int(cid), 0.0))
        ax.set_ylabel(f"Cluster {cid} | {pct:.1f}%", fontsize=8)
        ax.set_ylim(0, 1.05)

        x_min = np.nanmin(x) if start is None else start
        x_max = np.nanmax(x) if end is None else end
        ax.set_xlim(x_min, x_max)

        ax.grid(True, axis='x', linestyle='--',
                linewidth=0.5, color='gray')
        if show_pol2_line:
            ax.axvline(x=0, color="#C80028", linestyle="-", linewidth=2)

    # --- Shared x-axis formatting ---
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].xaxis.set_major_locator(MultipleLocator(100))
    axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(v)}"))
    axes[-1].get_xaxis().get_offset_text().set_visible(False)

    # --- Global labels ---
    fig.text(0.95, 0.5, 'Methylation level',
             va='center', ha='center', rotation=-90, fontsize=10)
    fig.text(0.05, 0.5, "Cluster number | Cluster proportion",
             fontsize=10, rotation=90, va='center', ha='center')
    fig.text(0.5, 0.04, "Genomic coordinate",
             va='center', ha='center', fontsize=10)

    plt.subplots_adjust(hspace=0.05, top=0.92)
    plt.setp(axes[-1].get_xticklabels(), rotation=45,
             ha='right', rotation_mode='anchor')
    axes[-1].tick_params(axis='x', labelsize=8)
    fig.subplots_adjust(bottom=0.18)

    plt.show()
    return fig, axes

if __name__ == "__main__":
    pass
