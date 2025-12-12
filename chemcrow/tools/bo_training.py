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
    name: str = "BayesianOptimizeData (ToyModel)"
    description: str = """Performs Bayesian optimization on numeric features from a CSV file.
Takes the csv_path, column names of the features as in the csv(like void_fraction, etc.), selectivity_Xe/Kr (target) value, and crsytal name (label) corresponding to the target, and finds the feature combination
that maximizes the target.
Takes:
    "path": path to CSV file
    "feature_cols": list of numeric feature column names
    "target_col": name of target column
    "label_col": optional label/identifier column
    "n_iter": number of optimization iterations (take 2 until user specify)
Returns top 10 candidates by Expected Improvement.
"""

    n_candidates: int = 50
    xi: float = 0.01

    def __init__(self, n_candidates: int = 50, xi: float = 0.01):
        super().__init__()
        self.n_candidates = n_candidates
        self.xi = xi

    def _parse_query(self, query):
        if isinstance(query, dict):
            return query
        if not isinstance(query, str):
            raise ValueError("Query must be a dict or string.")

        s = query.strip()
        if s.startswith("```") and s.endswith("```"):
            s = s[3:-3].strip()
        if s.startswith("`") and s.endswith("`"):
            s = s[1:-1].strip()

        try:
            return json.loads(s)
        except:
            try:
                return ast.literal_eval(s)
            except:
                start = s.find("{")
                end = s.rfind("}")
                if start != -1 and end != -1:
                    snippet = s[start:end+1]
                    try:
                        return json.loads(snippet)
                    except:
                        return ast.literal_eval(snippet)

        raise ValueError("Unable to parse query into dictionary.")

    def _expected_improvement(self, mu, sigma, best, xi=0.01):
        with np.errstate(divide="ignore", invalid="ignore"):
            imp = mu - best - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei = np.where(sigma <= 1e-12, np.maximum(0.0, imp), ei)
            return np.maximum(0.0, ei)

    def _run(self, query):

        try:
            q = self._parse_query(query)
        except Exception as e:
            return {"error": str(e)}

        # Load CSV
        path = q.get("path")
        if not path:
            return {"error": "CSV path is required."}

        try:
            df = pd.read_csv(path)
        except Exception as e:
            return {"error": f"Failed to read CSV: {e}"}

        feature_cols = q.get("feature_cols")
        if not feature_cols:
            return {"error": "'feature_cols' must be provided."}

        target_col = q.get("target_col")
        if not target_col:
            return {"error": "'target_col' must be provided."}

        label_col = q.get("label_col", None)

        # Extract data
        try:
            X = df[feature_cols].values.astype(float)
            y = df[target_col].values.astype(float)
        except Exception as e:
            return {"error": f"Failed to extract features/target: {e}"}

        labels = df[label_col].tolist() if label_col and label_col in df else [None] * len(df)

        if X.shape[0] != y.shape[0]:
            return {"error": "Mismatch between number of samples in X and y."}

        n_iter = int(q.get("n_iter", 2))

        # Standardise
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Gaussian Process
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(X.shape[1]),
                                           length_scale_bounds=(1e-2, 10))

        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            alpha=1e-6,
            normalize_y=True
        )

        print("\nBO Training started")

        gp.fit(X_scaled, y)

        # Bounds for sampling
        bounds = np.array([[X[:, i].min(), X[:, i].max()] for i in range(X.shape[1])])

        best_observed = float(np.max(y))
        all_candidates = []

        # Bayesian optimisation loop
        for _ in range(n_iter):
            candidates = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                           size=(self.n_candidates, X.shape[1]))

            candidates_scaled = scaler.transform(candidates)
            mu, sigma = gp.predict(candidates_scaled, return_std=True)
            ei = self._expected_improvement(mu, sigma, best_observed, xi=self.xi)

            # Store all candidates
            for i in range(len(candidates)):
                all_candidates.append({
                    "params": candidates[i],
                    "mu": float(mu[i]),
                    "sigma": float(sigma[i]),
                    "ei": float(ei[i])
                })

        # Select top 10 by EI
        all_candidates = sorted(all_candidates, key=lambda x: x["ei"], reverse=True)
        top_k = all_candidates[:10]

        # Match candidates to nearest observed point for label
        results = []
        for item in top_k:
            cand = item["params"]
            dists = np.linalg.norm(X - cand, axis=1)
            nearest_idx = int(np.argmin(dists))
            lbl = labels[nearest_idx]

            results.append({
                #"params": {feature_cols[j]: float(cand[j]) for j in range(len(feature_cols))},
                "predicted_value": item["mu"],
                #"ei": item["ei"],
                "label": lbl
            })

        return {"top_candidates": results}

    async def _arun(self, query):
        raise NotImplementedError("Async version not implemented.")
