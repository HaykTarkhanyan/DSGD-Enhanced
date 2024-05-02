import os
import logging
import datetime
import warnings

from pandas.errors import SettingWithCopyWarning

import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from config import *
from DSClassifierMultiQ import DSClassifierMultiQ


from utils import (calculate_adjusted_density, dbscan_predict,
                   detect_outliers_z_score, evaluate_classifier,
                   evaluate_clustering, filter_by_rule, get_distance,
                   get_kdist_plot, remove_outliers_and_normalize,
                   report_results, run_dbscan)

warnings.simplefilter(action='ignore', category=(SettingWithCopyWarning, FutureWarning, UserWarning))

# ---------------------------- logging ----------------------------

if not os.path.exists(LOG_FOLDER):
    os.mkdir(LOG_FOLDER)

log_file = os.path.join(LOG_FOLDER, "dst.log")

rfh = logging.handlers.RotatingFileHandler(
    filename=log_file,
    mode='a',
    maxBytes=LOGGING_MAX_SIZE_MB*1024*1024,
    backupCount=LOGGING_BACKUP_COUNT,
    encoding=None,
    delay=0
)

logging.getLogger('matplotlib.font_manager').disabled = True

console_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%y-%m-%d %H:%M:%S",
    handlers=[
        rfh,
        console_handler
    ],
)

logger = logging.getLogger(__name__)

class DSEnhanced:
    def __init__(self, dataset_folder: str, dataset: str, nrows: int = None, 
                 missing_threshold: float = MISSING_THRESHOLD, ratio_deviation: float=RATIO_DEVIATION, 
                 clustering_alg: str = None, train_set_size: float = TRAIN_SET_SIZE, 
                 eps: float = EPS, step: float = STEP, max_eps: float = MAX_EPS,
                 min_samples: int = None, target_clusters: int = TARGET_CLUSTERS, 
                 print_clustering_eval: bool = True, 
                 print_clustering_as_classification_eval: bool = True, 
                 mult_rules: bool = False, debug_mode: bool = True, print_final_model: bool = True,
                 num_breaks: int = 3, rules_folder: str = RULE_FOLDER, 
                 maf_methods: list = ["kmeans", "random", "uniform"]): 
        self.dataset_folder = dataset_folder
        assert os.path.exists(dataset_folder), "Dataset folder not found"

        self.datasets = os.listdir(dataset_folder)
        logging.info(f"Found {len(self.datasets)} datasets")
        self.dataset = dataset
        self.dataset_name = dataset.split(".")[0]
        self.nrows = nrows
        self.ratio_deviation = ratio_deviation
        self.missing_threshold = missing_threshold
        self.clustering_alg = clustering_alg
        self.train_set_size = train_set_size
        
        self.db_eps = None # for density based opacity calculation
        
        if clustering_alg == "dbscan":
            self.EPS = eps
            self.step = step
            self.max_eps = max_eps
            if min_samples:
                self.min_samples = min_samples # if not specified, will be assigned later 
            self.target_clusters = target_clusters
            
        self.print_clustering_eval = print_clustering_eval
        self.print_clustering_as_classification_eval = print_clustering_as_classification_eval
        
        # dst
        self.mult_rules = mult_rules
        self.debug_mode = debug_mode
        self.print_final_model = print_final_model
        self.num_breaks = num_breaks
        
        self.rules_folder = rules_folder
        
        self.maf_methods = maf_methods
    
    def read_data(self):
        if self.nrows:
            self.data_initial = pd.read_csv(os.path.join(self.dataset_folder, self.dataset), 
                                    nrows=self.nrows)
        else:
            self.data_initial = pd.read_csv(os.path.join(self.dataset_folder, self.dataset))
        
        logger.debug(f"Dataset: {self.dataset_name} | Shape: {self.data.shape}")

    def preprocess_data(self):
        self.data = self.data_initial.dropna(thresh=len(self.data) * (1 - self.missing_threshold), axis=1)
        logger.debug(f"{self.data.shape} - dropped columns with more than {self.missing_threshold*100:.0f}% missing values")
        self.data = self.data.dropna()
        logger.debug(f"{self.data.shape} - drop rows with missing values")
        
        assert self.data.isna().sum().sum() == 0, "Dataset contains missing values"
        assert "labels" in data.columns, "Dataset does not contain `labels` column"
        assert self.data.labels.nunique() == 2, f"Dataset labels are not binary ({self.data.labels.unique()})"

        label_ratio = self.data.labels.value_counts(normalize=True).iloc[0]
        assert abs(label_ratio -0.5) < self.ratio_deviation, f"Label ratio is not balanced ({label_ratio})"

        # leave only numeric columns
        data = data.select_dtypes(include=[np.number])
        logger.debug(f"{self.data.shape} drop non-numeric columns")

        # move labels column to the end 
        self.data = self.data[[col for col in self.data.columns if col != "labels"] + ["labels"]]

        logging.info(f"------ Dataset: {self.dataset_name} | Shape: {self.data.shape} | Label ratio: {label_ratio:.2f} -------")


    def train_test_split(self):
        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
        self.data = self.apply(pd.to_numeric)
        cut = int(train_set_size*len(self.data))

        self.train_data_df = self.data.iloc[:cut]
        self.test_data_df = self.data.iloc[cut:]

        self.X_train = self.data.iloc[:cut, :-1].values
        self.y_train = self.data.iloc[:cut, -1].values
        self.X_test = self.data.iloc[cut:, :-1].values
        self.y_test = self.data.iloc[cut:, -1].values

        logging.info(f"Step 0: Data split done | {len(self.X_train)} - {len(self.X_test)}")

    def standard_scaling(self):
        st_scaler = StandardScaler().fit(self.train_data_df)

        # TODO maybe delete this
        # scale = st_scaler.scale_
        # mean = st_scaler.mean_
        # var = st_scaler.var_ 

        self.X_train_scaled = st_scaler.transform(self.train_data_df)
        self.X_test_scaled = st_scaler.transform(self.test_data_df)  #! during inference we won't have this

        logging.debug("Step 1: Standard scaling complete")

    def clustering_and_inference(self):
        assert self.clustering_alg in ["kmeans", "dbscan"], "You must specify a clustering algorithm"   
        logging.info(f"Step 2.1: Performing {self.clustering_alg} clustering")

        if self.clustering_alg == "kmeans":
            self.clustering_model = KMeans(n_clusters=2, random_state=42, n_init="auto")      
            self.clustering_model.fit(self.X_train_scaled)  
            
            self.clustering_labels_train = self.clustering_model.predict(self.X_train_scaled)
            self.clustering_labels_test = self.clustering_model.predict(self.X_test_scaled)
        else:
            self.min_samples = 2 * self.X_train_scaled.shape[1] - 1
            self.clustering_model = run_dbscan(self.X_train_scaled, eps=self.EPS, 
                                               max_eps=self.max_eps, min_samples=self.min_samples, 
                                               step=self.step) 
            if self.clustering_model is None:
                logging.warning(f"Could not find the desired number of clusters for {self.dataset_name}")
                raise Exception("Clustering failed")
            
            self.clustering_labels_train = dbscan_predict(self.clustering_model, self.X_train_scaled)
            self.clustering_labels_test = dbscan_predict(self.clustering_model, self.X_test_scaled)
            
            self.db_eps = self.clustering_model.eps


        self.train_data_df["labels_clustering"] = self.clustering_labels_train
        self.test_data_df["labels_clustering"] = self.clustering_labels_test

        logger.info(f"Step 2.1: Clustering and inference done")
        
    def run_eval_clustering(self):
        logger.info("Step 2.2: Evaluate clustering")
        self.eval_clustering_train = evaluate_clustering(
            self.X_train_scaled, self.clustering_labels_train, self.clustering_model, 
            self.clustering_alg, print_results=self.print_clustering_eval, dataset="train")
        self.eval_clustering_test = evaluate_clustering(
            self.X_test_scaled, self.clustering_labels_test, self.clustering_model, 
            self.clustering_alg, print_results=self.print_clustering_eval, dataset="test")
        

    def run_eval_clustering_as_classifier(self):
        logging.info("Step 2.2: Clustering evaluation done")

        self.eval_clustering_as_classifier_train = evaluate_classifier(
            y_actual=self.y_train, y_clust=self.clustering_labels_train, 
            dataset="train", print_results=self.print_clustering_as_classification_eval)
        self.eval_clustering_as_classifier_test = evaluate_classifier(
            y_actual=self.y_test, y_clust=self.clustering_labels_test, 
            dataset="test", print_results=self.print_clustering_as_classification_eval)

        logger.info("Step 3: Clustering as a classifier, evaluation done")

    def get_opacity(self):
        self.train_data_df["distance"] = get_distance(
            self.X_train_scaled, self.clustering_model, 
            self.clustering_alg, density_radius=self.db_eps)
        self.test_data_df["distance"] = get_distance(
            self.X_test_scaled, self.clustering_model, 
            self.clustering_alg, density_radius=self.db_eps)

        self.train_data_df["distance_norm"] = remove_outliers_and_normalize(
            self.train_data_df) 
        self.test_data_df["distance_norm"] = remove_outliers_and_normalize(
            self.test_data_df)

        assert self.train_data_df.isna().sum().sum() == 0, "Train data contains NaNs"
        assert self.test_data_df.isna().sum().sum() == 0, "Train data contains NaNs"

        logger.info(f"Step 4: Opacity calculation done")
        
    def train_DST(self):
        ignore_for_training = ["labels_clustering", "distance_norm"]
        df_cols = [i for i in list(self.data.columns) if i not in ignore_for_training]

        logger.debug(f"Train: {len(self.X_train_use)}")

        for method in ["uniform"]: #["clustering", "random"]:
            name = f"dataset={self.dataset_name}, label_for_dist={LABEL_COL_FOR_DIST}, clust={self.clustering_alg}, breaks={self.num_breaks}, add_mult_rules={self.mult_rules}, maf_method={method}"
            logger.info(f"Step 5: Run DST ({name})")
            DSC = DSClassifierMultiQ(2, debug_mode=self.debug_mode, num_workers=self.num_workers, maf_method=method,
                                    data=self.train_data_df_use, precompute_rules=True, )#.head(rows_use))
            logger.debug(f"\tModel init done")    
            res = DSC.fit(self.X_train_use, self.y_train_use, 
                    add_single_rules=True, single_rules_breaks=self.num_breaks, add_mult_rules=self.mult_rules,
                    column_names=df_cols, print_every_epochs=1, print_final_model=self.print_final_model)
            losses, epoch, dt = res
            logger.debug(f"\tModel fit done")

            rules = DSC.model.save_rules_bin(os.path.join(self.rules_folder, f"{name}.dsb"))

            self.rules = DSC.model.find_most_important_rules()
            y_pred = DSC.predict(self.X_test)

            logger.info(f"Step 6: Inference done")

            self.dst_res = report_results(self.y_test, y_pred, dataset=self.dataset_name, method=method,
                        epoch=epoch, dt=dt, losses=losses, 
                        save_results=True, name=name, print_results=True,
                        breaks=self.num_breaks, mult_rules=self.mult_rules, 
                        clustering_alg=self.clustering_alg, label_for_dist=LABEL_COL_FOR_DIST)
            logging.info("-"*30)

    def save_all_important_res_to_json(self):
        logging.info("Step 7: Save all important results to json")
        for attr in ["eval_clustering_train", "eval_clustering_test", "eval_clustering_as_classifier_train", 
                     "eval_clustering_as_classifier_test", "opacity_train", "opacity_test", "dataset", 
                     "clustering_alg", "num_breaks", "mult_rules", "maf_methods", "train_data", "test_data", 
                     "rules", "X_train", "y_train", "X_test", "y_test", "X_train_scaled", "X_test_scaled", 
                     "clustering_labels_train", "clustering_labels_test", "db_eps", "train_data_df_use", 
                     "test_data_df_use", "X_train_use", "y_train_use", "X_test_use", "y_test_use", 
                     "train_data_df", "test_data_df", "data", "data_initial", "train_data_df_use", 
                     "test_data_df_use", "X_train_use", "y_train_use", "X_test_use", "y_test_use", 
                     "train_data_df", "test_data_df", "X_train_scaled", "X_test_scaled", 
                     "clustering_labels_train", "clustering_labels_test", "db_eps", "train_data_df_use", 
                     "test_data_df_use", "X_train_use", "y_train_use"]:
            self.dst_res[attr] = getattr(self, attr)
    
    def run(self):
        self.read_data()
        self.preprocess_data()
        self.train_test_split()
        self.standard_scaling()
        self.clustering_and_inference()
        self.run_eval_clustering()
        self.run_eval_clustering_as_classifier()
        self.get_opacity()
        self.train_DST()
        logging.info(f"Finished {self.dataset_name}")
        
    def run_all(self):
        for dataset in self.datasets:
            self.dataset = dataset
            self.run()
        logging.info("Finished all datasets")
    
    