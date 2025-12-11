import json
import ast
import numpy as np
import pandas as pd
from langchain.tools import BaseTool
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm

class BayesianOptimizeData(BaseTool):
    name: str = "BayesianOptimizeData"
    description: str = """Performs Bayesian optimization on numeric features from a CSV file.
Takes the csv_path, column names of the features as in the csv(like void_fraction, etc.), selectivity_Xe/Kr (target) value, and crsytal name (label) corresponding to the target, and finds the feature combination
that maximizes the target.

Input query (Python dict, JSON string, or Python-dict string) with keys:
    "path": "file.csv"             # path to CSV file
    "feature_cols": ["f1","f2",...]# feature columns name in CSV
    "target_col": "target"          # numeric target column in CSV
    "label_col": "label"            # optional human-level label column in CSV
    "n_iter": 2                    # number of optimization iterations (optional)
"""

    n_candidates: int = 50
    xi: float = 0.01

    def __init__(self, n_candidates: int = 50, xi: float = 0.01):
        super().__init__()
        self.n_candidates = n_candidates
        self.xi = xi

    def _parse_query(self, query):
        """Accept dict, JSON string, or Python dict string."""
        if isinstance(query, dict):
            return query
        if not isinstance(query, str):
            raise ValueError("Query must be a dict or string.")

        s = query.strip()
        # remove surrounding backticks or triple-backticks (copied from markdown/code blocks)
        if s.startswith('```') and s.endswith('```'):
            s = s[3:-3].strip()
        if s.startswith('`') and s.endswith('`'):
            s = s[1:-1].strip()

        # Try JSON first (must be valid JSON)
        try:
            return json.loads(s)
        except Exception as e_json:
            # Try python literal
            try:
                return ast.literal_eval(s)
            except Exception as e_ast:
                # Fallback: try to extract the first {...} substring which often contains the payload
                start = s.find('{')
                end = s.rfind('}')
                if start != -1 and end != -1 and end > start:
                    snippet = s[start:end+1]
                    try:
                        return json.loads(snippet)
                    except Exception:
                        try:
                            return ast.literal_eval(snippet)
                        except Exception:
                            pass

                # Nothing worked — raise a diagnostic error showing a safe snippet
                short = repr(s)[:300]
                raise ValueError(f"Unable to parse query. Expected dict or JSON/Python-dict string. "
                                 f"Parsing JSON failed: {e_json}; ast.literal_eval failed: {e_ast}. "
                                 f"Input snippet: {short}")

    def _expected_improvement(self, mu, sigma, best, xi=0.01):
        """Compute Expected Improvement for candidate points."""
        with np.errstate(divide='ignore', invalid='ignore'):
            imp = mu - best - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei = np.where(sigma <= 1e-12, np.maximum(0.0, imp), ei)
            return np.maximum(0.0, ei)

    def _run(self, query):

        try: 
            try:
                q = self._parse_query(query)
            except Exception as e:
                return {"error": str(e)}
    
            # --- Load CSV ---
            path = q.get("path")
            if not path:
                return {"error": "CSV path ('path') must be provided."}
            try:
                df = pd.read_csv(path)
            except Exception as e:
                return {"error": f"Failed to read CSV: {str(e)}"}
    
            feature_cols = q.get("feature_cols")
            if not feature_cols:
                return {"error": "'feature_cols' must be provided."}
            target_col = q.get("target_col")
            if not target_col:
                return {"error": "'target_col' must be provided."}
    
            try:
                X = df[feature_cols].values.astype(float)
                print("X = ")
                print(X.shape)
            except Exception as e:
                return Exception
    
            try:
                y = df[target_col].values.astype(float)
                print("Y = ")
                print(y.shape)
            except Exception as e:
                return Exception
    
            label_col = q.get("label_col")
            labels = df[label_col].tolist() if label_col in df else [None]*len(df)
    
            n_iter = int(q.get("n_iter", 2))
    
            # --- Validate ---
            if X.ndim != 2:
                return {"error": "X must be 2D (n_samples x n_features)."}
            if X.shape[0] != y.shape[0]:
                return {"error": "Number of rows in X must match length of y."}
            if len(labels) != X.shape[0]:
                return {"error": "Length of labels must match number of samples."}
    
            # --- Standardize ---
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
    
            # --- Fit Gaussian Process ---
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 10))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6, normalize_y=True)

            print("Training started")
            gp.fit(X_scaled, y)
    
            # --- Optimization loop ---
            bounds = np.array([[X[:, i].min(), X[:, i].max()] for i in range(X.shape[1])])
            best_val = float(np.max(y))
            best_point = None
    
            for _ in range(n_iter):
                candidates = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                               size=(self.n_candidates, X.shape[1]))
                candidates_scaled = scaler.transform(candidates)
                mu, sigma = gp.predict(candidates_scaled, return_std=True)
                ei = self._expected_improvement(mu, sigma, best_val, xi=self.xi)
                idx = int(np.argmax(ei))
                if ei[idx] > 0 and mu[idx] > best_val:
                    best_val = float(mu[idx])
                    best_point = candidates[idx]
    
            # --- Fallback to best observed row ---
            if best_point is None:
                best_idx = int(np.argmax(y))
                best_point = X[best_idx]
                best_val = float(y[best_idx])
    
            # --- Match best_point to observed row for label ---
            best_idx = None
            for i, row in enumerate(X):
                if np.allclose(row, best_point):
                    best_idx = i
                    break
            best_label = labels[best_idx] if best_idx is not None else None
    
            result = {feature_cols[i]: float(best_point[i]) for i in range(len(feature_cols))}
            return {"best_params": result, "predicted_max": best_val, "label": best_label}

        except Exception as e:
            print(e)

    async def _arun(self, query):
        raise NotImplementedError("Async run not implemented.")
