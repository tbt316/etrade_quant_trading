import numpy as np
import pandas as pd
from sklearn.decomposition import SparsePCA, PCA
from scipy.spatial.distance import cosine
import logging

logger = logging.getLogger(__name__)

class PCAFusion:
    def __init__(self, alpha=0.1, n_components=None):
        self.alpha = alpha
        self.n_components = n_components
        self.sparse_pca = None
        self.feature_names = None
        self.loadings_baseline = None

    def fit(self, df):
        """Fit Sparse PCA on the data."""
        self.feature_names = df.columns.tolist()
        
        # Determine optimal number of components if not provided
        if self.n_components is None:
            # Use standard PCA to find n_components for 90% variance
            pca = PCA()
            pca.fit(df)
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            self.n_components = max(2, np.argmax(cumulative_variance >= 0.90) + 1)
            logger.info(f"Automatically selected {self.n_components} components (min 2 enforced).")

        self.sparse_pca = SparsePCA(n_components=self.n_components, alpha=self.alpha, random_state=42)
        self.sparse_pca.fit(df)
        
        # FIX: Zero-Vector Fallback
        # If any component is all zeros, SparsePCA has collapsed. Fallback to Standard PCA.
        if np.any(np.all(self.sparse_pca.components_ == 0, axis=1)):
            logger.warning("⚠️ SparsePCA produced zero-vector components. Falling back to Standard PCA.")
            std_pca = PCA(n_components=self.n_components)
            std_pca.fit(df)
            self.sparse_pca = std_pca

        # Mandate 9.2: Eigenvector Sign Flipping
        if self.loadings_baseline is not None:
            for i in range(len(self.sparse_pca.components_)):
                # Calculate cosine similarity between current component and baseline
                sim = 1 - cosine(self.loadings_baseline[i], self.sparse_pca.components_[i])
                if sim < 0:
                    self.sparse_pca.components_[i] *= -1
                    logger.info(f"Flipped sign for PC{i+1} to maintain consistency.")

        self.loadings_baseline = self.sparse_pca.components_.copy()
        return self

    def transform(self, df):
        """Transform the data using the fitted Sparse PCA."""
        if self.sparse_pca is None:
            raise ValueError("PCAFusion must be fitted before calling transform.")
        pcs = self.sparse_pca.transform(df)
        
        # Defensive reshape for 1D returns or transposed results
        if pcs.ndim == 1:
            pcs = pcs.reshape(1, -1)
        elif pcs.shape[0] != len(df) and pcs.shape[1] == len(df):
            # Transposed return from some model types
            pcs = pcs.T
            
        pc_cols = [f"PC{i+1}" for i in range(self.n_components)]
        try:
            return pd.DataFrame(pcs, index=df.index, columns=pc_cols)
        except ValueError as e:
            logger.error(f"❌ PCA Shape Mismatch: pcs={pcs.shape}, index={len(df.index)}, columns={len(pc_cols)}")
            # Emergency fallback: just return what we have as a generic DF
            return pd.DataFrame(pcs, index=df.index)

    def fit_transform(self, df):
        """Fit Sparse PCA and transform the data with strict orthogonality enforcement."""
        self.fit(df)
        pcs = self.sparse_pca.transform(df)
        
        # Regulation 3.2: Strict Orthogonality Enforcement
        if pcs.shape[1] >= 2:
            pc_df_temp = pd.DataFrame(pcs)
            corr_matrix = pc_df_temp.corr().values
            np.fill_diagonal(corr_matrix, 0)
            max_corr = np.max(np.abs(corr_matrix))
            
            if max_corr > 0.1 or np.isnan(max_corr):
                logger.warning(f"⚠️ SparsePCA components non-orthogonal (max_corr={max_corr:.4f}). Falling back to Standard PCA.")
                std_pca = PCA(n_components=self.n_components)
                pcs = std_pca.fit_transform(df)
                self.sparse_pca = std_pca 
        
        pc_cols = [f"PC{i+1}" for i in range(self.n_components)]
        return pd.DataFrame(pcs, index=df.index, columns=pc_cols)

    def rolling_fit_transform(self, df, window=252):
        """
        Mandate 9.1: Sequential Subspace Fitting.
        Strictly causal rolling-window PCA. Fits on T-window and transforms only at T.
        """
        self.feature_names = df.columns.tolist()
        n_samples = len(df)
        
        # Determine n_components first if needed
        if self.n_components is None:
            pca = PCA()
            pca.fit(df.iloc[:min(window, n_samples)])
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            self.n_components = max(2, np.argmax(cumulative_variance >= 0.90) + 1)
            
        pc_values = np.full((n_samples, self.n_components), np.nan)

        for t in range(window, n_samples):
            train_window = df.iloc[t-window : t]
            self.fit(train_window)
            
            # Transform the single vector at time t
            target_vector = df.iloc[t : t+1]
            pc_values[t] = self.transform(target_vector).values[0]
            
        pc_cols = [f"PC{i+1}" for i in range(self.n_components)]
        return pd.DataFrame(pc_values, index=df.index, columns=pc_cols).dropna()

    def get_loadings_table(self):
        """Export the Sparse PCA loadings table for interpretability."""
        if self.sparse_pca is None:
            return pd.DataFrame()
        
        loadings = pd.DataFrame(
            self.sparse_pca.components_.T,
            index=self.feature_names,
            columns=[f"PC{i+1}" for i in range(self.n_components)]
        )
        return loadings

    def calculate_stability(self, current_df, window=10):
        """Calculate cosine similarity stability of loadings over a rolling window."""
        if len(current_df) < window:
            return 1.0
        
        recent_window = current_df.tail(window)
        recent_pca = SparsePCA(n_components=self.n_components, alpha=self.alpha, random_state=42)
        recent_pca.fit(recent_window)
        
        recent_loadings = recent_pca.components_
        
        # Compare each component's loading vector to the baseline
        similarities = []
        for i in range(self.n_components):
            sim = 1 - cosine(self.loadings_baseline[i], recent_loadings[i])
            similarities.append(sim)
            
        avg_stability = np.mean(similarities)
        if avg_stability < 0.7:
            logger.warning(f"🛑 STRUCTURAL BREAK DETECTED: Factor stability dropped to {avg_stability:.2f}")
            
        return avg_stability

    def verify_pca(self, original_df, pc_df):
        """🛑 Verification Checkpoint 2: Math Integrity (Strict)."""
        # 1. Strict Orthogonality Check
        corr_matrix = pc_df.corr().values
        np.fill_diagonal(corr_matrix, 0)
        max_corr = np.max(np.abs(corr_matrix))
        
        if max_corr > 0.1:
            raise ValueError(f"CRITICAL: PCA components failed orthogonality check: {max_corr:.4f} > 0.1 threshold.")
        
        # 2. Information Retention Check
        pca = PCA(n_components=self.n_components)
        pca.fit(original_df)
        retention = np.sum(pca.explained_variance_ratio_)
        
        if retention < 0.85:
            raise ValueError(f"PCA information retention too low: {retention:.2%}, target 85%+")
            
        logger.info(f"Verification Checkpoint 2 passed. Variance retention: {retention:.2%}")
        return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Dummy test
    data = pd.DataFrame(np.random.randn(100, 15), columns=[f"F{i}" for i in range(15)])
    fusion = PCAFusion()
    pcs = fusion.fit_transform(data)
    fusion.verify_pca(data, pcs)
    print("\n--- PC Loadings Table ---")
    print(fusion.get_loadings_table().round(2))
    print(f"\nStability: {fusion.calculate_stability(data):.4f}")
