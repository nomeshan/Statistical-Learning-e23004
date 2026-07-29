import json, io, base64, sys, contextlib, copy

SRC = '/mnt/user-data/uploads/Bayesian_Inference_Assignment__2_.ipynb'
nb = json.load(open(SRC))
cells = nb['cells']

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ns = {}
exec_count = [0]

def run_cell(source):
    """Execute code, capture stdout text output and any matplotlib figures as PNG outputs."""
    exec_count[0] += 1
    outputs = []
    buf = io.StringIO()
    plt.close('all')
    try:
        with contextlib.redirect_stdout(buf):
            exec(source, ns)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        outputs.append({
            "output_type": "error",
            "ename": type(e).__name__,
            "evalue": str(e),
            "traceback": tb.splitlines()
        })
        print(tb, file=sys.stderr)
        text = buf.getvalue()
        if text:
            outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": text.splitlines(keepends=True)})
        return outputs

    text = buf.getvalue()
    if text:
        outputs.append({"output_type": "stream", "name": "stdout", "text": text.splitlines(keepends=True)})

    fignums = plt.get_fignums()
    for n in fignums:
        fig = plt.figure(n)
        imgbuf = io.BytesIO()
        fig.savefig(imgbuf, format='png', dpi=110, bbox_inches='tight')
        imgbuf.seek(0)
        b64 = base64.b64encode(imgbuf.read()).decode('ascii')
        outputs.append({
            "output_type": "display_data",
            "data": {"image/png": b64, "text/plain": ["<Figure size matplotlib>"]},
            "metadata": {}
        })
    plt.close('all')
    return outputs

def make_code_cell(source, outputs):
    return {
        "cell_type": "code",
        "execution_count": exec_count[0],
        "metadata": {},
        "outputs": outputs,
        "source": source.splitlines(keepends=True)
    }

def make_md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True)
    }

# ---------------------------------------------------------------------------
# Replacement source for each code cell index (Plotly -> Matplotlib, offline)
# ---------------------------------------------------------------------------

CELL6 = r'''
import numpy as np
import matplotlib.pyplot as plt

# Define the 2PL Item Response Function
def p_i(theta, a, b):
    return 1 / (1 + np.exp(-a * (theta - b)))

# Generate a range of latent ability values (theta)
theta_vals = np.linspace(-6, 6, 300)

# Define configurations to plot
curves = [
    {"a": 0.5, "b": 0,  "style": "--"},
    {"a": 1.5, "b": -2, "style": "-"},
    {"a": 1.5, "b": 0,  "style": "-"},
    {"a": 1.5, "b": 2,  "style": "-"},
]

fig, ax = plt.subplots(figsize=(9, 5.5))
for curve in curves:
    a, b, style = curve["a"], curve["b"], curve["style"]
    p_vals = p_i(theta_vals, a, b)
    ax.plot(theta_vals, p_vals, style, linewidth=2.5, label=f"a = {a}, b = {b}")

ax.set_title("Two-Parameter Logistic (2PL) Item Response Curves", fontsize=13)
ax.set_xlabel("Latent Ability (θ)")
ax.set_ylabel("Probability of Correct Response P(Y_i = 1 | θ)")
ax.set_xlim(-6, 6)
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.25)
ax.legend(loc="upper left", framealpha=0.85)
plt.tight_layout()
plt.show()
'''

CELL12 = r'''
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# =====================================================================
# PART 1: SEQUENTIAL BAYESIAN UPDATE (MANUAL 4-ITEM SIMULATION)
# =====================================================================

theta = np.linspace(-5, 5, 500)
prior = stats.norm.pdf(theta, 0, 1)

def p_i(theta, a, b):
    return 1 / (1 + np.exp(-a * (theta - b)))

running_items = [
    {"a": 1.0, "b": -1.5, "y": 1},
    {"a": 1.5, "b": 0.5,  "y": 1},
    {"a": 1.2, "b": 1.5,  "y": 0},
    {"a": 2.0, "b": 0.2,  "y": 1}
]

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(theta, prior, '--', linewidth=2.5, color='gray', label='Initial Prior: N(0,1)')

current_posterior = prior.copy()
for idx, item in enumerate(running_items):
    a, b, y = item["a"], item["b"], item["y"]
    prob = p_i(theta, a, b)
    likelihood = (prob ** y) * ((1 - prob) ** (1 - y))
    current_posterior = current_posterior * likelihood
    integral = np.trapezoid(current_posterior, theta)
    current_posterior /= integral

    result_text = "Correct" if y == 1 else "Incorrect"
    trace_name = f"Step {idx+1}: After Item {idx+1} ({result_text}, a={a}, b={b})"
    ax1.plot(theta, current_posterior, linewidth=2, label=trace_name)

ax1.set_title("Sequential Bayesian Update of User Ability (θ)", fontsize=13)
ax1.set_xlabel("Latent Ability Parameter (θ)")
ax1.set_ylabel("Probability Density f(θ | y)")
ax1.grid(alpha=0.25)
ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.85)
plt.tight_layout()
plt.show()

# =====================================================================
# PART 2: PERFORMANCE TRACKING & CONVERGENCE TIMELINE (20 ITEMS)
# =====================================================================

np.random.seed(42)
theta_true = 0.75
n_items = 20
theta_grid = np.linspace(-5, 5, 1000)

a_params = np.random.uniform(0.5, 2.0, size=n_items)
b_params = np.random.normal(0, 1, size=n_items)

running_bayes = [0.0]
running_map = [0.0]
steps = list(range(n_items + 1))

current_posterior_sim = stats.norm.pdf(theta_grid, 0, 1)

for k in range(n_items):
    a_k, b_k = a_params[k], b_params[k]
    prob_true = p_i(theta_true, a_k, b_k)
    y_k = 1 if np.random.uniform(0, 1) < prob_true else 0

    prob_grid = p_i(theta_grid, a_k, b_k)
    likelihood = (prob_grid ** y_k) * ((1 - prob_grid) ** (1 - y_k))

    current_posterior_sim = current_posterior_sim * likelihood
    integral_sim = np.trapezoid(current_posterior_sim, theta_grid)
    current_posterior_sim /= integral_sim

    theta_bayes_k = np.trapezoid(theta_grid * current_posterior_sim, theta_grid)
    theta_map_k = theta_grid[np.argmax(current_posterior_sim)]

    running_bayes.append(theta_bayes_k)
    running_map.append(theta_map_k)

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.axhline(theta_true, linestyle='--', color='red', linewidth=2,
            label=f"True Ability (θ = {theta_true})")
ax2.plot(steps, running_bayes, marker='o', color='blue', linewidth=2.5,
         label='Posterior Mean (θ̂_Bayes)')
ax2.plot(steps, running_map, marker='s', color='green', linewidth=2,
         label='MAP Estimate (θ̂_MAP)')

ax2.set_title("Convergence of Latent Ability Estimators (θ) Over Time", fontsize=13)
ax2.set_xlabel("Sequence / Item Position (k)")
ax2.set_ylabel("Estimated Ability (θ̂)")
ax2.set_xticks(range(0, n_items + 1, 2))
ax2.set_ylim(-1, 2)
ax2.grid(alpha=0.25)
ax2.legend(loc="lower left", framealpha=0.9)
plt.tight_layout()
plt.show()

print(f"Final Posterior-Mean estimate after {n_items} items: theta_hat_Bayes = {running_bayes[-1]:.4f}")
print(f"Final MAP estimate after {n_items} items:            theta_hat_MAP   = {running_map[-1]:.4f}")
print(f"True ability used to simulate responses:              theta_true      = {theta_true}")
'''

CELL17 = r'''
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

theta_grid = np.linspace(0, 1, 500)

beta_configs = [
    {"alpha": 1, "beta": 1, "name": "Uninformative State: Beta(1,1)", "color": "gray",  "style": "--"},
    {"alpha": 2, "beta": 8, "name": "Right-Skewed State: Beta(2,8)",  "color": "blue",  "style": "-"},
    {"alpha": 8, "beta": 2, "name": "Left-Skewed State: Beta(8,2)",   "color": "green", "style": "-"}
]

fig, ax = plt.subplots(figsize=(9, 5.5))
for config in beta_configs:
    a, b = config["alpha"], config["beta"]
    pdf_vals = stats.beta.pdf(theta_grid, a, b)
    ax.plot(theta_grid, pdf_vals, config["style"], color=config["color"],
            linewidth=2.5, label=config["name"])

ax.set_title("Structural Variations of the Beta(α, β) Probability Density Function", fontsize=12.5)
ax.set_xlabel("Parameter Value (θ)")
ax.set_ylabel("Probability Density f(θ)")
ax.set_xlim(0, 1)
ax.grid(alpha=0.25)
ax.legend(loc="upper center", framealpha=0.85)
plt.tight_layout()
plt.show()
'''

CELL23 = r'''
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

np.random.seed(42)

theta_true = 0.35
n_impressions = 100
steps = list(range(n_impressions + 1))

alpha_param = 1
beta_param = 1

theta_grid = np.linspace(0, 1, 500)
milestones = [0, 1, 2, 5, 10, 30, 50, 100]

running_bayes = [alpha_param / (alpha_param + beta_param)]
running_map = [0.0]

fig1, ax1 = plt.subplots(figsize=(10, 6))
prior_density = stats.beta.pdf(theta_grid, alpha_param, beta_param)
ax1.plot(theta_grid, prior_density, '--', linewidth=2.5, color='gray',
         label='Initial Prior: Beta(1,1)')

for k in range(1, n_impressions + 1):
    y_k = 1 if np.random.uniform(0, 1) < theta_true else 0
    alpha_param += y_k
    beta_param += (1 - y_k)

    theta_bayes_k = alpha_param / (alpha_param + beta_param)
    if alpha_param > 1 and beta_param > 1:
        theta_map_k = (alpha_param - 1) / (alpha_param + beta_param - 2)
    else:
        theta_map_k = 0.0 if alpha_param <= beta_param else 1.0

    running_bayes.append(theta_bayes_k)
    running_map.append(theta_map_k)

    if k in milestones:
        density_k = stats.beta.pdf(theta_grid, alpha_param, beta_param)
        result_text = "Click" if y_k == 1 else "No Click"
        ax1.plot(theta_grid, density_k, linewidth=2,
                 label=f"Step {k}: After Event ({result_text}, α={alpha_param}, β={beta_param})")

ax1.axvline(theta_true, linestyle=':', color='red', linewidth=2,
            label=f"True CTR ({theta_true})")
ax1.set_title("Analytical Posterior Density Progression (Beta-Binomial Updates)", fontsize=12.5)
ax1.set_xlabel("Conversion Rate Parameter (θ)")
ax1.set_ylabel("Probability Density f(θ | y)")
ax1.grid(alpha=0.25)
ax1.legend(loc="upper right", fontsize=8, framealpha=0.85)
plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.axhline(theta_true, linestyle='--', color='red', linewidth=2,
            label=f"True CTR (θ = {theta_true})")
ax2.plot(steps, running_bayes, color='blue', linewidth=2.5,
         label='Exact Posterior Mean (Beta Formula)')
ax2.plot(steps, running_map, color='green', linewidth=1.5, linestyle=':',
         label='Exact MAP Estimate (Beta Formula)')

ax2.set_title("Analytical Beta-Binomial Conjugate Update Timeline", fontsize=12.5)
ax2.set_xlabel("Number of User Impressions (k)")
ax2.set_ylabel("Estimated Conversion Rate (θ̂)")
ax2.grid(alpha=0.25)
ax2.legend(loc="lower right", framealpha=0.9)
plt.tight_layout()
plt.show()

print(f"Final posterior after {n_impressions} impressions: Beta(alpha={alpha_param}, beta={beta_param})")
print(f"Final Bayes (posterior-mean) estimate: {running_bayes[-1]:.4f}")
print(f"Final MAP estimate:                    {running_map[-1]:.4f}")
print(f"True CTR:                              {theta_true}")
'''

CELL28 = r'''
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

np.random.seed(24)

theta_true = 0.68
K_nominal = 50.0
sigma = 0.15
n_sensor_readings = 15

theta_grid = np.linspace(0.01, 1.0, 500)

current_posterior = stats.beta.pdf(theta_grid, a=8, b=1.5)
current_posterior /= np.trapezoid(current_posterior, theta_grid)

milestones = [0, 1, 2, 5, 10, 15]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(theta_grid, current_posterior, '--', linewidth=2.5, color='gray',
        label='Prior State: Structural Health Assumed Healthy')

for k in range(1, n_sensor_readings + 1):
    noise = np.random.normal(0, sigma)
    y_k = (theta_true * K_nominal) * np.exp(noise)

    expected_K = theta_grid * K_nominal
    likelihood = stats.lognorm.pdf(y_k, s=sigma, scale=expected_K)

    current_posterior = current_posterior * likelihood
    integral = np.trapezoid(current_posterior, theta_grid)
    current_posterior /= integral

    if k in milestones:
        ax.plot(theta_grid, current_posterior, linewidth=2,
                label=f"Step {k}: Post-Sensor Reading (Observed K={y_k:.2f})")

ax.axvline(theta_true, linestyle=':', color='red', linewidth=2.5,
           label=f"True Structural Degradation State ({theta_true})")

ax.set_title("Structural Health Monitoring: Bounded Bayesian Parameter Updating", fontsize=12.5)
ax.set_xlabel("Remaining Structural Stiffness Efficiency Factor (θ)")
ax.set_ylabel("Probability Density (Confidence level of damage)")
ax.grid(alpha=0.25)
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.85)
plt.tight_layout()
plt.show()

theta_bayes_final = np.trapezoid(theta_grid * current_posterior, theta_grid)
theta_map_final = theta_grid[np.argmax(current_posterior)]
print(f"After {n_sensor_readings} sensor readings:")
print(f"  Posterior-mean estimate of stiffness efficiency: {theta_bayes_final:.4f}")
print(f"  MAP estimate of stiffness efficiency:             {theta_map_final:.4f}")
print(f"  True stiffness efficiency:                        {theta_true}")
'''

# --- Part 10: replace Kaggle-download cells with an offline synthetic proxy dataset ---

NOTE_MD_PART10 = (
    "**Note on data source:** the original exercise pulls the "
    "`arjunbhasin2013/ccdata` Kaggle credit-card dataset via `kagglehub`. "
    "This execution environment has no internet access, so a synthetic "
    "proxy dataset with the same two feature columns used by the model "
    "(`PURCHASES`, `CREDIT_LIMIT`) is generated instead, built from a "
    "3-component mixture so the GMM has genuine structure to recover. "
    "The `GMMFinancialSegmenter` class and its logic are unchanged from the "
    "original Kaggle-based design — only the data-loading step differs."
)

CELL_DATA = r'''
import numpy as np
import pandas as pd

np.random.seed(42)
n1, n2, n3 = 400, 300, 100

synthetic_data = {
    "PURCHASES": np.hstack([
        np.random.exponential(400, n1),
        np.random.normal(2500, 600, n2),
        np.random.normal(6000, 1200, n3),
    ]),
    "CREDIT_LIMIT": np.hstack([
        np.random.normal(2000, 800, n1),
        np.random.normal(7000, 1500, n2),
        np.random.normal(12000, 2000, n3),
    ]),
}
synthetic_data["PURCHASES"] = np.clip(synthetic_data["PURCHASES"], 0, None)
synthetic_data["CREDIT_LIMIT"] = np.clip(synthetic_data["CREDIT_LIMIT"], 100, None)

df2 = pd.DataFrame(synthetic_data)
print("Synthetic credit-card-style dataset (proxy for arjunbhasin2013/ccdata):")
print(df2.shape)
df2.head()
'''

CELL56 = r'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class GMMFinancialSegmenter:

    def __init__(self, n_components=3, random_state=42):
        """Initializes the GMM Segmenter framework."""
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            random_state=self.random_state,
        )

    def prepare_data(self, df, feature_cols, test_size=0.2):
        """Extracts features, normalizes them, and splits into train/test sets."""
        X = df[feature_cols].dropna().values
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test = train_test_split(
            X_scaled, test_size=test_size, random_state=self.random_state
        )
        return X_train, X_test

    def fit(self, X_train):
        """Trains the Gaussian Mixture Model on training data."""
        self.model.fit(X_train)
        print(" GMM Training complete.")
        print(f"Converged: {self.model.converged_}")
        print(f"Iterations taken: {self.model.n_iter_}")

    def evaluate(self, X_test):
        """Validates the model on test data using average log-likelihood."""
        avg_log_likelihood = self.model.score(X_test)
        print(f"\n Validation Performance (Test Set):")
        print(f"Average Log-Likelihood: {avg_log_likelihood:.4f}")
        return avg_log_likelihood

    def plot_density_heatmap(self, X_train, feature_names):
        """Generates a 2D density heatmap of the empirical training distribution."""
        X_orig = self.scaler.inverse_transform(X_train)

        fig = plt.figure(figsize=(8, 8))
        gs = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05)
        ax_main = fig.add_subplot(gs[1:4, 0:3])
        ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

        hb = ax_main.hist2d(X_orig[:, 0], X_orig[:, 1], bins=35, cmap="viridis")
        ax_main.set_xlabel(feature_names[0])
        ax_main.set_ylabel(feature_names[1])

        ax_top.hist(X_orig[:, 0], bins=35, color="steelblue")
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.set_ylabel("count")

        ax_right.hist(X_orig[:, 1], bins=35, orientation="horizontal", color="steelblue")
        ax_right.tick_params(axis="y", labelleft=False)
        ax_right.set_xlabel("count")

        fig.suptitle("Empirical Training Data Density Heatmap", fontsize=13, y=0.95)
        plt.show()

    def _generate_contour_base(self, X_data):
        """Helper to compute coordinate grids and posterior probability maps."""
        x_min, x_max = X_data[:, 0].min() - 0.5, X_data[:, 0].max() + 0.5
        y_min, y_max = X_data[:, 1].min() - 0.5, X_data[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200)
        )

        grid_points = np.c_[xx.ravel(), yy.ravel()]
        responsibilities = self.model.predict_proba(grid_points)
        max_prob = responsibilities.max(axis=1).reshape(xx.shape)

        grid_orig = self.scaler.inverse_transform(grid_points)
        xx_orig = grid_orig[:, 0].reshape(xx.shape)
        yy_orig = grid_orig[:, 1].reshape(yy.shape)

        return xx_orig, yy_orig, max_prob

    def _plot_assignments(self, X, feature_names, title, label_prefix):
        xx_orig, yy_orig, max_prob = self._generate_contour_base(X)
        hard_labels = self.model.predict(X)
        X_orig = self.scaler.inverse_transform(X)

        fig, ax = plt.subplots(figsize=(9, 7))
        cf = ax.contourf(xx_orig, yy_orig, max_prob, levels=20, cmap="cividis", alpha=0.65)
        fig.colorbar(cf, ax=ax, label="Max Responsibility (Confidence)")

        for k in range(self.n_components):
            mask = hard_labels == k
            ax.scatter(X_orig[mask, 0], X_orig[mask, 1], s=22,
                       edgecolors="black", linewidths=0.4,
                       label=f"{label_prefix} Cluster {k+1}")

        ax.set_title(title, fontsize=12.5)
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.legend(loc="best", framealpha=0.85)
        plt.tight_layout()
        plt.show()

    def plot_training_assignments(self, X_train, feature_names):
        """Plots the training data points over the soft assignment confidence boundaries."""
        self._plot_assignments(
            X_train, feature_names,
            "GMM Soft-Assignment Confidence Boundaries on Training Data",
            "Train"
        )

    def plot_soft_assignments(self, X_test, feature_names):
        """Plots the test data points overlaying the continuous soft assignment profiles."""
        self._plot_assignments(
            X_test, feature_names,
            "GMM Soft-Assignment Confidence Boundaries on Test Data",
            "Test"
        )


# =====================================================================
# Execution Block: Using synthetic credit-card-style data columns
# (offline proxy for the Kaggle arjunbhasin2013/ccdata dataset)
# =====================================================================
if __name__ == "__main__":
    df = df2
    features = ["PURCHASES", "CREDIT_LIMIT"]

    segmenter = GMMFinancialSegmenter(n_components=3)
    X_train, X_test = segmenter.prepare_data(df, features)
    segmenter.fit(X_train)
    segmenter.evaluate(X_test)

    segmenter.plot_density_heatmap(X_train, features)
    segmenter.plot_training_assignments(X_train, features)
    segmenter.plot_soft_assignments(X_test, features)
'''

replacements = {
    6: CELL6,
    12: CELL12,
    17: CELL17,
    23: CELL23,
    28: CELL28,
    56: CELL56,
}

# indices to drop entirely (old kaggle loading cells 51,52,53,54; keep 55 blank md maybe drop)
drop_indices = {51, 52, 53, 54}

new_cells = []
for i, c in enumerate(cells):
    if i in drop_indices:
        continue
    if c['cell_type'] == 'markdown':
        new_cells.append(c)
    else:
        if i in replacements:
            src = replacements[i].strip('\n') + '\n'
            outs = run_cell(src)
            new_cells.append(make_code_cell(src, outs))
        else:
            new_cells.append(c)

    if i == 50:
        new_cells.append(make_md_cell(NOTE_MD_PART10))
        src = CELL_DATA.strip('\n') + '\n'
        outs = run_cell(src)
        new_cells.append(make_code_cell(src, outs))

nb['cells'] = new_cells

# Set kernelspec / metadata for portability
nb.setdefault('metadata', {})
nb['metadata']['kernelspec'] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
}
nb['metadata']['language_info'] = {"name": "python", "version": "3.11"}

out_path = '/home/claude/work/Bayesian_Inference_Assignment_Completed.ipynb'
with open(out_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("DONE ->", out_path, "cells:", len(new_cells))
