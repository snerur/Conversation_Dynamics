"""
Conversational Dynamics in Embedding Space
A Streamlit implementation of the proof-of-concept study design from the memo:
  "Conversational Dynamics in Embedding Space: A Theoretical Framework
   and Proof-of-Concept Design"

Five core measurements (Section 8.2):
  1. Estimate g — conditional-mean kernel + conditional variance σ²(E)
  2. Validate Markov approximation — 1-step vs 2-step prediction
  3. Lyapunov exponent — divergence from paired trajectories
  4. Stationary distribution — concentration and attractor structure
  5. Periodic structure — DMD spectrum + recurrence plots

Baselines (Section 8.3):
  - Shuffled conversation data
  - Random walk on sphere (noise dynamics)
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
import streamlit as st

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Conversational Dynamics Explorer",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌀 Conversational Dynamics in Embedding Space")
st.caption(
    "Study multi-turn LLM conversations as discrete-time dynamical systems. "
    "Based on the theoretical framework in *conversation_dynamics_memo.pdf*."
)

# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "conversations": [],        # List[{turns, embeddings, seed}]
    "paired_conv": None,        # {conv_A, conv_B} for Lyapunov
    "encoder": None,
    "encoder_name": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — Configuration
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # ── LLM ──────────────────────────────────────────────────────────────────
    st.subheader("Language Model")

    PROVIDER_MODELS = {
        "openai":    ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022",
                      "claude-opus-4-5"],
        "google":    ["gemini/gemini-1.5-flash", "gemini/gemini-1.5-pro",
                      "gemini/gemini-2.0-flash"],
        "ollama":    ["ollama/llama3.2", "ollama/llama3.1", "ollama/mistral",
                      "ollama/gemma2", "ollama/qwen2.5"],
        "other":     ["custom"],
    }
    PROVIDER_ENV_KEYS = {
        "openai":    "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google":    "GOOGLE_API_KEY",
    }

    provider = st.selectbox("Provider", list(PROVIDER_MODELS.keys()))
    model_choices = PROVIDER_MODELS[provider]
    model_name = st.selectbox("Model", model_choices)
    if model_name == "custom":
        model_name = st.text_input("Custom model name (litellm format)",
                                   placeholder="e.g. together_ai/mistralai/Mixtral-8x7B")

    api_key = st.text_input(
        "API Key", type="password",
        help="Leave blank for Ollama or if key is already in environment"
    )
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
    max_tokens  = st.number_input("Max tokens per turn", 30, 600, 150, step=10)

    st.divider()

    # ── Encoder ───────────────────────────────────────────────────────────────
    st.subheader("Sentence Encoder")

    enc_backend = st.radio(
        "Encoder backend",
        ["OpenAI API", "Local (sentence-transformers)"],
        help="OpenAI API requires an API key but avoids local model downloads. "
             "Local runs offline but needs compatible huggingface_hub.",
    )

    if enc_backend == "OpenAI API":
        OPENAI_ENC_MODELS = {
            "text-embedding-3-small  (1536-dim · fast · cheap)": "text-embedding-3-small",
            "text-embedding-3-large  (3072-dim · best quality)": "text-embedding-3-large",
            "text-embedding-ada-002  (1536-dim · legacy)":       "text-embedding-ada-002",
        }
        enc_display  = st.selectbox("OpenAI embedding model", list(OPENAI_ENC_MODELS.keys()))
        enc_model_id = OPENAI_ENC_MODELS[enc_display]
        enc_api_key  = st.text_input(
            "OpenAI API key for embeddings", type="password",
            help="Leave blank to reuse the LLM API key above (only works if provider = openai).",
        )

        if st.button("Activate Encoder"):
            _key = enc_api_key or (api_key if provider == "openai" else "")
            if not _key:
                st.error("An OpenAI API key is required. Enter it above or set provider to openai and enter it in the LLM section.")
            else:
                with st.spinner("Testing connection …"):
                    try:
                        from openai import OpenAI as _OAI
                        _client = _OAI(api_key=_key)
                        _client.embeddings.create(input=["ping"], model=enc_model_id)
                        st.session_state.encoder = {
                            "backend": "openai",
                            "model":   enc_model_id,
                            "api_key": _key,
                        }
                        st.session_state.encoder_name = f"openai/{enc_model_id}"
                        st.success(f"Ready: {enc_model_id}")
                    except Exception as exc:
                        st.error(f"OpenAI embedding error: {exc}")

    else:  # Local sentence-transformers
        LOCAL_ENC_MODELS = {
            "all-MiniLM-L6-v2  (fast · 384-dim)":           "all-MiniLM-L6-v2",
            "all-mpnet-base-v2  (better · 768-dim)":         "all-mpnet-base-v2",
            "paraphrase-MiniLM-L6-v2  (paraphrase-tuned)":  "paraphrase-MiniLM-L6-v2",
            "all-distilroberta-v1  (384-dim)":               "all-distilroberta-v1",
        }
        enc_display  = st.selectbox("Local model", list(LOCAL_ENC_MODELS.keys()))
        enc_model_id = LOCAL_ENC_MODELS[enc_display]

        if st.button("Load Encoder"):
            with st.spinner(f"Downloading / loading {enc_model_id} …"):
                try:
                    from sentence_transformers import SentenceTransformer
                    _model = SentenceTransformer(enc_model_id)
                    st.session_state.encoder = {"backend": "local", "model": _model}
                    st.session_state.encoder_name = enc_model_id
                    st.success(f"Loaded: {enc_model_id}")
                except Exception as exc:
                    st.error(f"Failed to load encoder: {exc}")
                    st.info(
                        "If you see a `HfFolder` import error, fix it with:  \n"
                        "`pip install -U sentence-transformers huggingface_hub`  \n"
                        "Or switch to the **OpenAI API** backend above."
                    )

    if st.session_state.encoder_name:
        st.info(f"Active: **{st.session_state.encoder_name}**")
    else:
        st.warning("No encoder loaded — activate one above before generating.")

    st.divider()
    if st.button("🗑️ Clear all conversation data"):
        st.session_state.conversations = []
        st.session_state.paired_conv   = None
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Helper: LLM call
# ──────────────────────────────────────────────────────────────────────────────
def call_llm(messages: list) -> str | None:
    """Thin wrapper around litellm.completion."""
    try:
        import litellm
        litellm.set_verbose = False

        if api_key and provider in PROVIDER_ENV_KEYS:
            os.environ[PROVIDER_ENV_KEYS[provider]] = api_key

        resp = litellm.completion(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=int(max_tokens),
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        st.error(f"LLM error: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Helper: Self-conversation
# ──────────────────────────────────────────────────────────────────────────────
def run_self_conversation(
    seed: str,
    system_prompt: str,
    n_turns: int,
    status_placeholder=None,
) -> list[dict] | None:
    """
    Run an LLM self-conversation:
      Turn n user input  = assistant output from Turn n-1
      Turn 1 user input  = seed
    Returns list of {turn, input, output}.
    """
    messages: list[dict] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})

    turns: list[dict] = []
    current_input = seed

    for i in range(n_turns):
        if status_placeholder:
            status_placeholder.write(f"Turn {i + 1} / {n_turns} …")

        messages.append({"role": "user", "content": current_input})
        response = call_llm(messages)
        if response is None:
            break

        messages.append({"role": "assistant", "content": response})
        turns.append({"turn": i + 1, "input": current_input, "output": response})
        current_input = response   # ← self-conversation: feed output back as input

    return turns or None


# ──────────────────────────────────────────────────────────────────────────────
# Helper: Embed conversation turns
# ──────────────────────────────────────────────────────────────────────────────
def embed_turns(turns: list[dict]) -> np.ndarray | None:
    """Embed each turn's output and L2-normalise to unit sphere.

    Supports two backends stored in st.session_state.encoder:
      {"backend": "openai", "model": str, "api_key": str}
      {"backend": "local",  "model": SentenceTransformer}
    """
    if st.session_state.encoder is None:
        st.error("Please activate a sentence encoder first (sidebar).")
        return None

    enc   = st.session_state.encoder
    texts = [t["output"] for t in turns]

    if enc["backend"] == "openai":
        try:
            from openai import OpenAI as _OAI
            client  = _OAI(api_key=enc["api_key"])
            # OpenAI accepts up to 2048 inputs per call; batch if needed
            BATCH   = 100
            vectors = []
            for i in range(0, len(texts), BATCH):
                resp = client.embeddings.create(
                    input=texts[i : i + BATCH],
                    model=enc["model"],
                )
                vectors.extend([d.embedding for d in resp.data])
            emb = np.array(vectors, dtype=np.float32)
        except Exception as exc:
            st.error(f"OpenAI embedding error: {exc}")
            return None
    else:  # local sentence-transformers
        try:
            emb = enc["model"].encode(texts, normalize_embeddings=True)
            emb = np.array(emb, dtype=np.float32)
        except Exception as exc:
            st.error(f"Local encoder error: {exc}")
            return None

    # L2-normalise to unit sphere (OpenAI already returns normalised, but be safe)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb   = emb / np.clip(norms, 1e-9, None)
    return emb   # shape (T, d)


# ──────────────────────────────────────────────────────────────────────────────
# Analysis functions
# ──────────────────────────────────────────────────────────────────────────────

def step_cosine_distances(emb: np.ndarray) -> np.ndarray:
    """Cosine distance between consecutive embeddings (T-1 values)."""
    return np.array([1.0 - float(np.dot(emb[i], emb[i + 1]))
                     for i in range(len(emb) - 1)])


def recurrence_matrix(emb: np.ndarray) -> np.ndarray:
    """Full pairwise cosine-distance matrix (T × T)."""
    sim = emb @ emb.T
    return 1.0 - np.clip(sim, -1.0, 1.0)


def estimate_kernel_variance(all_emb: list[np.ndarray]) -> dict | None:
    """
    Fit the conditional-mean kernel g(E) = E[E_{n+1} | E_n] using k-NN
    and measure the residual + conditional variance σ²(E).
    """
    Xs, Ys = [], []
    for emb in all_emb:
        for i in range(len(emb) - 1):
            Xs.append(emb[i])
            Ys.append(emb[i + 1])
    if len(Xs) < 8:
        return None

    X, Y = np.array(Xs), np.array(Ys)
    k = max(3, min(8, len(X) // 3))
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, indices = nbrs.kneighbors(X)

    residuals, variances = [], []
    for i in range(len(X)):
        nb_idx    = indices[i][1:]          # exclude self
        nb_Y      = Y[nb_idx]               # successors of neighbours
        g_hat     = nb_Y.mean(axis=0)
        norm      = np.linalg.norm(g_hat)
        g_hat     = g_hat / norm if norm > 1e-9 else g_hat

        residuals.append(1.0 - float(np.dot(g_hat, Y[i])))
        var = np.mean(
            [1.0 - float(np.clip(np.dot(s / np.linalg.norm(s), g_hat), -1, 1))
             for s in nb_Y if np.linalg.norm(s) > 1e-9]
        )
        variances.append(var)

    return {
        "residuals":     np.array(residuals),
        "variances":     np.array(variances),
        "mean_residual": float(np.mean(residuals)),
        "mean_variance": float(np.mean(variances)),
    }


def markov_test(all_emb: list[np.ndarray]) -> dict | None:
    """
    Markov test: does (E_{n-1}, E_n) predict E_{n+1} better than E_n alone?
    Returns errors and improvement percentage.
    """
    X1s, X2s, Ys = [], [], []
    for emb in all_emb:
        if len(emb) < 3:
            continue
        for i in range(1, len(emb) - 1):
            X1s.append(emb[i])
            X2s.append(np.concatenate([emb[i - 1], emb[i]]))
            Ys.append(emb[i + 1])

    if len(X1s) < 12:
        return None

    X1, X2, Y = np.array(X1s), np.array(X2s), np.array(Ys)
    rng  = np.random.default_rng(42)
    idx  = rng.permutation(len(X1))
    cut  = int(0.8 * len(idx))
    tr, te = idx[:cut], idx[cut:]

    k = max(3, min(7, len(tr) // 4))

    def _cos_err(pred: np.ndarray, true: np.ndarray) -> float:
        norms = np.linalg.norm(pred, axis=1, keepdims=True)
        pred_n = pred / np.clip(norms, 1e-9, None)
        return float(np.mean(1.0 - np.einsum("ij,ij->i", pred_n, true)))

    knn1 = KNeighborsRegressor(n_neighbors=k).fit(X1[tr], Y[tr])
    knn2 = KNeighborsRegressor(n_neighbors=k).fit(X2[tr], Y[tr])

    err1 = _cos_err(knn1.predict(X1[te]), Y[te])
    err2 = _cos_err(knn2.predict(X2[te]), Y[te])
    improvement = (err1 - err2) / err1 * 100.0 if err1 > 1e-9 else 0.0

    return {"err_1step": err1, "err_2step": err2,
            "improvement_pct": improvement, "n_samples": len(X1)}


def compute_lyapunov(
    emb_A: np.ndarray,
    emb_B: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Estimate Lyapunov exponent λ from two paired trajectories.
    Fits d(t) ≈ d(0) · exp(λ t) in log space.
    λ > 0 → diverging (sensitive / chaotic-like)
    λ < 0 → converging (contracting toward attractor)
    """
    n = min(len(emb_A), len(emb_B))
    dists = np.array([
        max(1.0 - float(np.dot(emb_A[i], emb_B[i])), 1e-9)
        for i in range(n)
    ])
    log_d  = np.log(dists)
    turns  = np.arange(n, dtype=float)
    coeffs = np.polyfit(turns, log_d, 1)
    λ      = coeffs[0]
    fit    = np.polyval(coeffs, turns)
    return dists, log_d, λ, fit


def compute_dmd(emb: np.ndarray, r: int = 10) -> np.ndarray | None:
    """
    SVD-based Dynamic Mode Decomposition.
    Returns complex eigenvalues of the reduced propagator.
    |λ| ≈ 1 → oscillatory mode; < 1 → decaying; > 1 → growing.
    """
    if len(emb) < 4:
        return None
    X  = emb[:-1].T    # d × (T-1)
    Xp = emb[1:].T     # d × (T-1)
    r  = min(r, X.shape[0] - 1, X.shape[1] - 1)
    if r < 2:
        return None

    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    Ur, Sr, Vr = U[:, :r], S[:r], Vh[:r, :].T
    A_tilde = Ur.T @ Xp @ Vr @ np.diag(1.0 / Sr)
    return np.linalg.eigvals(A_tilde)


def pca_project(emb_list: list[np.ndarray], n_comp: int = 3):
    """Fit PCA on all embeddings concatenated; return projections + var ratio."""
    all_emb = np.vstack(emb_list)
    n       = min(n_comp, all_emb.shape[1], all_emb.shape[0] - 1)
    pca     = PCA(n_components=n)
    proj    = pca.fit_transform(all_emb)
    return proj, pca.explained_variance_ratio_, pca


# ──────────────────────────────────────────────────────────────────────────────
# Baseline generators
# ──────────────────────────────────────────────────────────────────────────────

def shuffled_baseline(conversations: list[dict]) -> list[np.ndarray]:
    """Shuffle embeddings within each conversation — destroys temporal order."""
    out = []
    rng = np.random.default_rng(0)
    for conv in conversations:
        e = conv["embeddings"].copy()
        rng.shuffle(e)
        out.append(e)
    return out


def noise_baseline(conversations: list[dict]) -> list[np.ndarray]:
    """Random walk on the sphere with the same mean step size as the real data."""
    rng  = np.random.default_rng(1)
    out  = []
    for conv in conversations:
        emb       = conv["embeddings"]
        avg_step  = step_cosine_distances(emb).mean()
        d         = emb.shape[1]
        traj      = [emb[0].copy()]
        for _ in range(len(emb) - 1):
            noise     = rng.standard_normal(d)
            noise    /= np.linalg.norm(noise)
            scale     = np.sqrt(2.0 * avg_step)   # approximate geodesic step
            candidate = traj[-1] + scale * noise
            traj.append(candidate / np.linalg.norm(candidate))
        out.append(np.array(traj, dtype=np.float32))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────
_COLORS = px.colors.qualitative.Set1


def fig_trajectory_2d(
    emb_list: list[np.ndarray],
    labels: list[str] | None = None,
) -> go.Figure:
    all_emb  = np.vstack(emb_list)
    n_comp   = min(2, all_emb.shape[1], all_emb.shape[0] - 1)
    pca      = PCA(n_components=n_comp).fit(all_emb)
    var      = pca.explained_variance_ratio_

    fig    = go.Figure()
    offset = 0
    for ci, emb in enumerate(emb_list):
        n    = len(emb)
        proj = pca.transform(emb)
        lbl  = labels[ci] if labels else f"Conv {ci + 1}"
        col  = _COLORS[ci % len(_COLORS)]
        pc2  = proj[:, 1] if n_comp > 1 else np.zeros(n)

        fig.add_trace(go.Scatter(
            x=proj[:, 0], y=pc2, mode="lines+markers", name=lbl,
            line=dict(color=col, width=1.8),
            marker=dict(size=7, color=list(range(n)),
                        colorscale="Blues", showscale=False),
            text=[f"Turn {i + 1}" for i in range(n)],
            hovertemplate="<b>%{text}</b><br>PC1=%{x:.3f} PC2=%{y:.3f}",
        ))
        # Mark start
        fig.add_trace(go.Scatter(
            x=[proj[0, 0]], y=[pc2[0]], mode="markers", showlegend=False,
            marker=dict(symbol="star", size=14, color=col),
        ))
        offset += n

    fig.update_layout(
        title=f"Embedding Trajectory — PCA 2D  (PC1 {var[0]:.1%} var"
              + (f", PC2 {var[1]:.1%} var)" if n_comp > 1 else ")"),
        xaxis_title="PC1", yaxis_title="PC2",
        template="plotly_white", hovermode="closest",
    )
    return fig


def fig_step_distances(conversations: list[dict]) -> go.Figure:
    fig = go.Figure()
    for i, conv in enumerate(conversations):
        dists = step_cosine_distances(conv["embeddings"])
        fig.add_trace(go.Scatter(
            x=list(range(1, len(dists) + 1)), y=dists,
            mode="lines+markers", name=f"Conv {i + 1}",
            line=dict(color=_COLORS[i % len(_COLORS)]),
        ))
    fig.update_layout(
        title="Step-Size: Cosine Distance Between Consecutive Turns",
        xaxis_title="Turn", yaxis_title="Cosine Distance",
        template="plotly_white",
    )
    return fig


def fig_recurrence(emb: np.ndarray, title: str = "Recurrence Plot") -> go.Figure:
    R = recurrence_matrix(emb)
    n = len(emb)
    fig = go.Figure(go.Heatmap(
        z=R,
        x=[f"T{i + 1}" for i in range(n)],
        y=[f"T{i + 1}" for i in range(n)],
        colorscale="RdBu_r", zmin=0, zmax=1,
        colorbar=dict(title="Cos dist"),
    ))
    fig.update_layout(title=title, xaxis_title="Turn", yaxis_title="Turn",
                      template="plotly_white")
    return fig


def fig_dmd_spectrum(eigenvalues: np.ndarray) -> go.Figure:
    theta = np.linspace(0, 2 * np.pi, 200)
    fig   = go.Figure()
    # Unit circle
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta), mode="lines",
        line=dict(color="gray", dash="dot"), name="Unit circle",
    ))
    mag = np.abs(eigenvalues)
    fig.add_trace(go.Scatter(
        x=eigenvalues.real, y=eigenvalues.imag,
        mode="markers", name="DMD eigenvalues",
        marker=dict(size=11, color=mag, colorscale="Viridis",
                    colorbar=dict(title="|λ|"), showscale=True,
                    line=dict(width=1, color="white")),
        text=[f"|λ|={m:.3f}  freq={np.angle(e) / (2 * np.pi):.3f} cyc/turn"
              for e, m in zip(eigenvalues, mag)],
        hovertemplate="<b>%{text}</b><br>Re=%{x:.3f}  Im=%{y:.3f}",
    ))
    fig.update_layout(
        title="DMD Eigenvalue Spectrum (Koopman approximation)",
        xaxis_title="Re(λ)", yaxis_title="Im(λ)",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        template="plotly_white",
    )
    return fig


def fig_lyapunov(
    dists: np.ndarray, log_d: np.ndarray, λ: float, fit: np.ndarray
) -> go.Figure:
    turns = list(range(len(dists)))
    fig   = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Distance Between Paired Trajectories",
                        "Log-Distance + Linear Fit (Lyapunov)"],
    )
    fig.add_trace(go.Scatter(x=turns, y=dists, mode="lines+markers",
                             name="d(t)", line=dict(color="steelblue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=turns, y=log_d, mode="markers",
                             name="log d(t)", marker=dict(color="steelblue")), row=1, col=2)
    fig.add_trace(go.Scatter(x=turns, y=fit, mode="lines",
                             name=f"Fit λ={λ:.4f}", line=dict(color="crimson", dash="dash")),
                  row=1, col=2)
    direction = "↑ diverging" if λ > 0 else "↓ converging"
    fig.update_layout(
        title=f"Lyapunov Analysis  —  λ ≈ {λ:.4f}  ({direction})",
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Turn")
    fig.update_yaxes(title_text="Cosine Distance", row=1, col=1)
    fig.update_yaxes(title_text="log(Cosine Distance)", row=1, col=2)
    return fig


def fig_stationary(all_emb_list: list[np.ndarray]) -> go.Figure:
    proj, var, _ = pca_project(all_emb_list, n_comp=2)
    labels = []
    for i, e in enumerate(all_emb_list):
        labels.extend([i + 1] * len(e))

    fig = go.Figure(go.Scatter(
        x=proj[:, 0],
        y=proj[:, 1] if proj.shape[1] > 1 else np.zeros(len(proj)),
        mode="markers",
        marker=dict(size=6, color=labels, colorscale="Turbo",
                    colorbar=dict(title="Conv #"), opacity=0.75),
        hovertemplate="PC1=%{x:.3f} PC2=%{y:.3f}",
    ))
    fig.update_layout(
        title=f"Stationary Distribution (all turns) — {sum(var[:2]):.1%} variance explained",
        xaxis_title=f"PC1 ({var[0]:.1%})",
        yaxis_title=f"PC2 ({var[1]:.1%})" if len(var) > 1 else "PC2",
        template="plotly_white",
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab_gen, tab_traj, tab_dyn, tab_base = st.tabs([
    "🗣️ Generate Conversations",
    "📈 Trajectory Explorer",
    "🔬 Dynamical Analysis",
    "📊 Baselines",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Generate Conversations
# ═══════════════════════════════════════════════════════════════════════════════
with tab_gen:
    st.header("Generate Self-Conversations")
    st.info(
        "**Self-conversation**: at each turn the LLM's previous response becomes the next "
        "user message. This creates a discrete-time trajectory in embedding space — the "
        "central object of study in the paper."
    )

    col_left, col_right = st.columns([3, 2])

    with col_left:
        system_prompt = st.text_area(
            "System prompt / persona",
            value=(
                "You are a curious and reflective AI engaged in an open-ended philosophical "
                "dialogue. Keep each response to 2–4 sentences: thought-provoking but concise."
            ),
            height=90,
        )
        seed_topic = st.text_input(
            "Seed topic (Turn 1 user message)",
            value="What is the nature of consciousness, and could it ever emerge in an artificial mind?",
        )
        c1, c2, c3 = st.columns(3)
        n_turns         = c1.number_input("Turns per conversation", 5, 60, 15, step=5)
        n_conversations = c2.number_input("# conversations", 1, 30, 3)
        run_paired      = c3.checkbox(
            "Paired trajectory\n(for Lyapunov)",
            help="Also run a second conversation from a slightly different seed to measure divergence.",
        )

    with col_right:
        perturbed_seed = st.text_area(
            "Perturbed seed (for paired / Lyapunov run)",
            value="What is the essence of consciousness, and could it ever arise within an artificial mind?",
            height=90,
            disabled=not run_paired,
            help="A slight rephrasing of the seed. The Lyapunov exponent measures how fast the "
                 "two trajectories diverge.",
        )
        st.caption(
            "Tip: change just a word or two so the initial embeddings are close "
            "but not identical — otherwise the divergence signal is noisy from the start."
        )

    if st.button("🚀 Run Conversations", type="primary"):
        if st.session_state.encoder is None:
            st.error("Load a sentence encoder in the sidebar first.")
            st.stop()

        prog      = st.progress(0.0, "Starting…")
        status_ph = st.empty()
        new_convs = []

        for ci in range(n_conversations):
            prog.progress(ci / n_conversations, f"Conversation {ci + 1}/{n_conversations}")
            with st.spinner(f"Running conversation {ci + 1} …"):
                turns = run_self_conversation(
                    seed=seed_topic,
                    system_prompt=system_prompt,
                    n_turns=int(n_turns),
                    status_placeholder=status_ph,
                )
                if turns:
                    emb = embed_turns(turns)
                    if emb is not None:
                        new_convs.append({
                            "turns":      turns,
                            "embeddings": emb,
                            "seed":       seed_topic,
                        })

        prog.progress(1.0, "Done!")

        # Paired trajectory
        if run_paired and new_convs:
            with st.spinner("Running paired conversation for Lyapunov analysis …"):
                turns_B = run_self_conversation(
                    seed=perturbed_seed,
                    system_prompt=system_prompt,
                    n_turns=int(n_turns),
                )
                if turns_B:
                    emb_B = embed_turns(turns_B)
                    if emb_B is not None:
                        st.session_state.paired_conv = {
                            "conv_A": new_convs[0],
                            "conv_B": {
                                "turns":      turns_B,
                                "embeddings": emb_B,
                                "seed":       perturbed_seed,
                            },
                        }

        st.session_state.conversations.extend(new_convs)
        status_ph.empty()
        st.success(
            f"Added {len(new_convs)} conversation(s). "
            f"Total in session: **{len(st.session_state.conversations)}**"
            + (" | Paired conversation stored for Lyapunov." if run_paired and st.session_state.paired_conv else "")
        )

    # ── Show stored conversations ─────────────────────────────────────────────
    if st.session_state.conversations:
        st.divider()
        st.subheader(f"Stored Conversations  ({len(st.session_state.conversations)})")
        for ci, conv in enumerate(st.session_state.conversations):
            with st.expander(f"Conversation {ci + 1}  ·  {len(conv['turns'])} turns  ·  seed: {conv['seed'][:60]}…"):
                for t in conv["turns"]:
                    st.markdown(f"**↳ Turn {t['turn']}**")
                    st.markdown(f"*Input:* {t['input']}")
                    st.markdown(f"*Output:* {t['output']}")
                    st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Trajectory Explorer
# ═══════════════════════════════════════════════════════════════════════════════
with tab_traj:
    st.header("Embedding Trajectory Explorer")

    if not st.session_state.conversations:
        st.warning("Generate conversations in Tab 1 first.")
        st.stop()

    all_emb = [c["embeddings"] for c in st.session_state.conversations]

    # PCA 2D
    st.subheader("Trajectory in PCA Space")
    st.plotly_chart(fig_trajectory_2d(all_emb), use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Step Distances")
        st.plotly_chart(fig_step_distances(st.session_state.conversations),
                        use_container_width=True)
        all_dists = np.concatenate([step_cosine_distances(e) for e in all_emb])
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Mean step", f"{all_dists.mean():.4f}")
        mc2.metric("Std step",  f"{all_dists.std():.4f}")
        mc3.metric("Max step",  f"{all_dists.max():.4f}")

    with col_b:
        st.subheader("Recurrence Plot")
        conv_sel = st.selectbox(
            "Conversation", range(len(st.session_state.conversations)),
            format_func=lambda i: f"Conv {i + 1}",
            key="rec_sel",
        )
        st.plotly_chart(
            fig_recurrence(
                st.session_state.conversations[conv_sel]["embeddings"],
                f"Recurrence Plot — Conv {conv_sel + 1}  "
                "(diagonal bands = periodic; solid blocks = stuck)",
            ),
            use_container_width=True,
        )

    # Summary table
    st.subheader("Per-Conversation Statistics")
    rows = []
    for i, conv in enumerate(st.session_state.conversations):
        emb   = conv["embeddings"]
        dists = step_cosine_distances(emb)
        rows.append({
            "Conv": i + 1,
            "Turns": len(emb),
            "Mean step dist": f"{dists.mean():.4f}",
            "Std step dist":  f"{dists.std():.4f}",
            "Total distance": f"{dists.sum():.4f}",
            "Drift from start (mean)": f"{np.mean(1 - emb @ emb[0]):.4f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Dynamical Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab_dyn:
    st.header("Dynamical Systems Analysis")
    st.caption("Five measurements ordered from foundational to downstream (Section 8.2).")

    if not st.session_state.conversations:
        st.warning("Generate conversations in Tab 1 first.")
        st.stop()

    all_emb = [c["embeddings"] for c in st.session_state.conversations]

    # ── 1. Kernel estimation ──────────────────────────────────────────────────
    st.subheader("① Kernel Estimation — g(E) and σ²(E)")
    with st.expander("What is this?", expanded=False):
        st.markdown(
            "Fits the conditional-mean kernel **g(E) = E[E_{n+1} | E_n]** using k-NN regression "
            "on observed (E_n, E_{n+1}) pairs.  "
            "The **conditional variance σ²(E)** is the framework's core viability diagnostic: "
            "small σ² → the embedding is sufficient; large σ² → information is being lost."
        )

    kv = estimate_kernel_variance(all_emb)
    if kv is None:
        st.warning("Need at least 8 (E_n, E_{n+1}) pairs. Run more conversations.")
    else:
        qA, qB, qC = st.columns(3)
        qA.metric("Mean prediction residual", f"{kv['mean_residual']:.4f}",
                  help="Average cosine distance between g(E_n) and the true E_{n+1}")
        qB.metric("Mean conditional variance σ²", f"{kv['mean_variance']:.4f}",
                  help="Spread of successors given the same neighbourhood in E-space")

        quality = ("✅ Tight" if kv["mean_variance"] < 0.05
                   else "⚠️ Moderate" if kv["mean_variance"] < 0.15
                   else "❌ Loose — embedding may be losing dynamics-relevant info")
        qC.metric("Framework quality", quality)

        fig_kv = go.Figure()
        fig_kv.add_trace(go.Histogram(x=kv["residuals"], name="Prediction residual",
                                      opacity=0.7, nbinsx=25, marker_color="steelblue"))
        fig_kv.add_trace(go.Histogram(x=kv["variances"], name="σ²(E)",
                                      opacity=0.7, nbinsx=25, marker_color="coral"))
        fig_kv.update_layout(barmode="overlay", template="plotly_white",
                             title="Residual and Conditional Variance Distribution",
                             xaxis_title="Cosine Distance", yaxis_title="Count")
        st.plotly_chart(fig_kv, use_container_width=True)

    st.divider()

    # ── 2. Markov test ────────────────────────────────────────────────────────
    st.subheader("② Markov Approximation Test")
    with st.expander("What is this?", expanded=False):
        st.markdown(
            "Compares k-NN prediction of **E_{n+1}** from **E_n alone** (1-step) vs from "
            "**(E_{n-1}, E_n)** (2-step).  \n"
            "If the 2-step predictor is appreciably better, the dynamics has memory and "
            "lag-embedding is needed before further analysis."
        )

    total_pairs = sum(max(0, len(e) - 1) for e in all_emb)
    if total_pairs < 20:
        st.warning(f"Only {total_pairs} transition pairs — run more conversations for a reliable test.")
    else:
        mr = markov_test(all_emb)
        if mr is None:
            st.warning("Not enough data yet.")
        else:
            mA, mB, mC = st.columns(3)
            mA.metric("1-step error", f"{mr['err_1step']:.4f}")
            mB.metric("2-step error", f"{mr['err_2step']:.4f}")
            mC.metric("Improvement from history", f"{mr['improvement_pct']:.1f}%")

            if mr["improvement_pct"] > 10:
                st.warning(
                    f"**Non-Markovian signal**: history improves prediction by "
                    f"{mr['improvement_pct']:.1f}%. Lag-embedding is recommended."
                )
            elif mr["improvement_pct"] > 3:
                st.info(
                    f"**Weak memory**: history improves by {mr['improvement_pct']:.1f}%. "
                    "Markov approximation is reasonable but imperfect."
                )
            else:
                st.success(
                    f"**Markov approximation holds**: history adds only "
                    f"{mr['improvement_pct']:.1f}% improvement. "
                    "Embedding is approximately a sufficient statistic."
                )

            fig_mr = go.Figure(go.Bar(
                x=["1-step  (E_n → E_{n+1})", "2-step  ([E_{n-1}, E_n] → E_{n+1})"],
                y=[mr["err_1step"], mr["err_2step"]],
                marker_color=["steelblue", "coral"],
            ))
            fig_mr.update_layout(
                title="Markov Test — Prediction Error Comparison",
                yaxis_title="Mean Cosine Distance to True E_{n+1}",
                template="plotly_white",
            )
            st.plotly_chart(fig_mr, use_container_width=True)

    st.divider()

    # ── 3. Lyapunov exponent ──────────────────────────────────────────────────
    st.subheader("③ Lyapunov Exponent")
    with st.expander("What is this?", expanded=False):
        st.markdown(
            "Tracks the cosine distance between two trajectories started from slightly "
            "different seeds.  \n"
            "Fits **d(t) ≈ d₀ · e^{λt}** in log-space.  \n"
            "- **λ > 0** → trajectories diverge: sensitive dependence on initial conditions.  \n"
            "- **λ < 0** → trajectories converge: contracting toward an attractor.  \n"
            "Generate a *paired conversation* in Tab 1 to enable this analysis."
        )

    if st.session_state.paired_conv is None:
        st.warning(
            "No paired conversation stored. "
            "Check *'Paired trajectory (for Lyapunov)'* in Tab 1 before running."
        )
    else:
        pc   = st.session_state.paired_conv
        eA   = pc["conv_A"]["embeddings"]
        eB   = pc["conv_B"]["embeddings"]
        dists, log_d, λ, fit = compute_lyapunov(eA, eB)

        lA, lB = st.columns(2)
        lA.metric("Estimated λ", f"{λ:.4f}")
        lB.metric("Direction", "Diverging ↑" if λ > 0 else "Converging ↓")

        st.plotly_chart(fig_lyapunov(dists, log_d, λ, fit), use_container_width=True)

        seed_A = pc["conv_A"]["seed"]
        seed_B = pc["conv_B"]["seed"]
        st.caption(
            f"**Seed A**: {seed_A}  \n"
            f"**Seed B**: {seed_B}  \n"
            "Note: with short conversations the estimate can be noisy. "
            "More turns and multiple pairs improve reliability."
        )

    st.divider()

    # ── 4. Stationary distribution ────────────────────────────────────────────
    st.subheader("④ Stationary Distribution")
    with st.expander("What is this?", expanded=False):
        st.markdown(
            "Where do long conversations concentrate in embedding space?  \n"
            "**Tight concentration** → mode collapse (strong attractor).  \n"
            "**Broad spread** → diverse dynamics.  \n"
            "**Multiple clusters** → multiple attractors / basins."
        )

    st.plotly_chart(fig_stationary(all_emb), use_container_width=True)

    all_emb_cat = np.vstack(all_emb)
    pca3        = PCA(n_components=min(3, all_emb_cat.shape[1], all_emb_cat.shape[0] - 1))
    proj3       = pca3.fit_transform(all_emb_cat)

    sA, sB, sC = st.columns(3)
    sA.metric("PCA variance explained (top-3 PCs)",
              f"{sum(pca3.explained_variance_ratio_):.1%}")
    sB.metric("Spread — std in PCA space", f"{proj3.std():.4f}")
    n_s   = min(150, len(all_emb_cat))
    idx_s = np.random.default_rng(7).choice(len(all_emb_cat), n_s, replace=False)
    samp  = all_emb_cat[idx_s]
    pw    = 1.0 - samp @ samp.T
    sC.metric("Mean pairwise cosine dist", f"{pw[np.triu_indices(n_s, k=1)].mean():.4f}")

    st.divider()

    # ── 5. Periodic structure ─────────────────────────────────────────────────
    st.subheader("⑤ Periodic Structure — DMD Spectrum + Recurrence Plot")
    with st.expander("What is this?", expanded=False):
        st.markdown(
            "**Dynamic Mode Decomposition** extracts coherent spatial-temporal modes.  \n"
            "Each eigenvalue encodes a frequency (angle) and growth/decay rate (magnitude).  \n"
            "- On the unit circle → stable oscillation (periodic mode).  \n"
            "- Inside → decaying.  Outside → growing.  \n\n"
            "**Recurrence plots**: diagonal line patterns → periodic; "
            "complex speckle → chaotic; solid blocks → mode collapse."
        )

    dmd_conv_sel = st.selectbox(
        "Conversation for DMD", range(len(st.session_state.conversations)),
        format_func=lambda i: f"Conv {i + 1}", key="dmd_sel",
    )
    emb_dmd = st.session_state.conversations[dmd_conv_sel]["embeddings"]
    max_rank = max(2, min(20, len(emb_dmd) - 2))
    dmd_r    = st.slider("DMD rank (number of modes)", 2, max_rank,
                         min(10, max_rank), key="dmd_r")

    eigs = compute_dmd(emb_dmd, r=dmd_r)
    if eigs is None:
        st.warning("Need at least 4 turns for DMD.")
    else:
        st.plotly_chart(fig_dmd_spectrum(eigs), use_container_width=True)
        mag  = np.abs(eigs)
        dA, dB, dC = st.columns(3)
        dA.metric("Modes near unit circle (0.9 < |λ| < 1.1)",
                  f"{sum(0.9 < m < 1.1 for m in mag)} / {len(mag)}")
        dB.metric("Mean |λ|", f"{mag.mean():.4f}")
        freqs = np.abs(np.angle(eigs)) / (2 * np.pi)
        dC.metric("Dominant frequency (cycles/turn)", f"{freqs.max():.4f}")

        st.plotly_chart(
            fig_recurrence(emb_dmd, f"Recurrence Plot — Conv {dmd_conv_sel + 1}"),
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Baselines
# ═══════════════════════════════════════════════════════════════════════════════
with tab_base:
    st.header("Baselines and Null Models")
    st.caption(
        "Any genuine dynamical structure in the real conversations should stand out clearly "
        "against these null models (Section 8.3)."
    )

    if not st.session_state.conversations:
        st.warning("Generate conversations in Tab 1 first.")
        st.stop()

    all_emb   = [c["embeddings"] for c in st.session_state.conversations]
    shuf_emb  = shuffled_baseline(st.session_state.conversations)
    noise_emb = noise_baseline(st.session_state.conversations)
    real_dists  = np.concatenate([step_cosine_distances(e) for e in all_emb])
    shuf_dists  = np.concatenate([step_cosine_distances(e) for e in shuf_emb])
    noise_dists = np.concatenate([step_cosine_distances(e) for e in noise_emb])

    # ── Shuffled ──────────────────────────────────────────────────────────────
    st.subheader("Shuffled Baseline (temporal null)")
    st.caption(
        "Embeddings are randomly reordered within each conversation, destroying temporal "
        "structure. Any real dynamical signal should differ."
    )

    col1, col2 = st.columns(2)
    with col1:
        fig_hist_shuf = go.Figure()
        for data, name, col in [
            (real_dists,  "Real",     "steelblue"),
            (shuf_dists,  "Shuffled", "coral"),
        ]:
            fig_hist_shuf.add_trace(go.Histogram(
                x=data, name=name, opacity=0.7, nbinsx=25, marker_color=col))
        fig_hist_shuf.update_layout(barmode="overlay", template="plotly_white",
                                    title="Step Distances: Real vs Shuffled",
                                    xaxis_title="Cosine Distance", yaxis_title="Count")
        st.plotly_chart(fig_hist_shuf, use_container_width=True)

    with col2:
        # Kernel variance comparison
        real_kv  = estimate_kernel_variance(all_emb)
        shuf_kv  = estimate_kernel_variance(shuf_emb)
        if real_kv and shuf_kv:
            fig_kv_cmp = go.Figure(go.Bar(
                x=["Real", "Shuffled"],
                y=[real_kv["mean_variance"], shuf_kv["mean_variance"]],
                marker_color=["steelblue", "coral"],
            ))
            fig_kv_cmp.update_layout(
                title="Mean Conditional Variance σ²: Real vs Shuffled",
                yaxis_title="Mean σ²(E)",
                template="plotly_white",
            )
            st.plotly_chart(fig_kv_cmp, use_container_width=True)
        st.caption(
            "Shuffled data should have **higher** σ² if real conversations have genuine "
            "temporal structure."
        )

    st.divider()

    # ── Noise (random walk) ───────────────────────────────────────────────────
    st.subheader("Random Walk Baseline (noise null)")
    st.caption(
        "A random walk on the sphere with the same mean step size as the real data. "
        "Real conversations should show more structured trajectories."
    )

    col3, col4 = st.columns(2)
    with col3:
        fig_traj_cmp = fig_trajectory_2d(
            all_emb + noise_emb,
            labels=(
                [f"Real {i + 1}" for i in range(len(all_emb))]
                + [f"Noise {i + 1}" for i in range(len(noise_emb))]
            ),
        )
        st.plotly_chart(fig_traj_cmp, use_container_width=True)

    with col4:
        fig_hist_noise = go.Figure()
        for data, name, col in [
            (real_dists,  "Real",        "steelblue"),
            (noise_dists, "Random Walk", "green"),
        ]:
            fig_hist_noise.add_trace(go.Histogram(
                x=data, name=name, opacity=0.7, nbinsx=25, marker_color=col))
        fig_hist_noise.update_layout(barmode="overlay", template="plotly_white",
                                     title="Step Distances: Real vs Random Walk",
                                     xaxis_title="Cosine Distance", yaxis_title="Count")
        st.plotly_chart(fig_hist_noise, use_container_width=True)

    # Recurrence comparison
    st.subheader("Recurrence Plots: Real vs Noise")
    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(
            fig_recurrence(all_emb[0], "Real — Conversation 1"),
            use_container_width=True,
        )
    with col6:
        st.plotly_chart(
            fig_recurrence(noise_emb[0], "Random Walk Baseline"),
            use_container_width=True,
        )

    st.divider()

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader("Baseline Comparison Summary")
    noise_kv = estimate_kernel_variance(noise_emb)

    summary_rows = []
    for label, dists_arr, kv_res in [
        ("Real conversations",  real_dists,  real_kv),
        ("Shuffled (null)",     shuf_dists,  shuf_kv),
        ("Random walk (null)",  noise_dists, noise_kv),
    ]:
        summary_rows.append({
            "Dataset": label,
            "Mean step dist":   f"{dists_arr.mean():.4f}",
            "Std step dist":    f"{dists_arr.std():.4f}",
            "Mean pred residual": f"{kv_res['mean_residual']:.4f}" if kv_res else "N/A",
            "Mean σ²(E)":        f"{kv_res['mean_variance']:.4f}" if kv_res else "N/A",
        })

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    st.caption(
        "**Interpretation**: Real conversations should have lower σ²(E) than shuffled data "
        "(temporal structure matters) and different step-distance distributions vs the random walk."
    )
