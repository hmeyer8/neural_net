"""Central configuration for the RFC agent.

Every tunable that affects a reported number lives here, and `Settings.fingerprint()`
hashes the ones that do. That hash goes in the `config_hash` column of
`experiments/experiments.csv`, so a row in the ledger can be tied back to the exact
configuration that produced it. A number I cannot reproduce is not a result.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The global seed. Everything stochastic in this package derives from it:
# train/dev/test splits, any sampling in the eval harness, provider temperature
# where the provider honors a seed.
SEED = 1337


def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw).expanduser() if raw else default


def resolve_device(preference: str = "auto") -> str:
    """Turn a device preference into a concrete torch device string.

    "auto" means cuda when torch reports a usable GPU, cpu otherwise. Anything
    else is taken literally, so RFCAGENT_DEVICE=cpu forces the CPU path even on a
    machine that has a GPU.

    Resolved once and passed down rather than re-checked at each call site: a
    system that silently runs half on GPU and half on CPU because two modules
    asked the question separately is a system whose latency numbers mean nothing.
    """
    if preference != "auto":
        return preference
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass
    return "cpu"


def device_report() -> str:
    """One human-readable line describing what the encoders will actually run on.

    Printed by `rfcagent status`. The point is that "is this using the GPU" should
    never be a thing anyone has to guess at, because the answer silently changes
    every latency number in the eval report.
    """
    device = resolve_device(settings.retrieval.device)
    if device != "cuda":
        return f"{device} (torch reports no usable CUDA device)"
    try:
        import torch

        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"cuda — {name}, {total:.1f} GB, torch {torch.__version__}"
    except Exception as exc:  # pragma: no cover - diagnostic path only
        return f"cuda (could not query device: {exc})"


@dataclass(frozen=True)
class Paths:
    """Where things live on disk.

    `data/` and `artifacts/` are both gitignored. The corpus is ~9,700 public-domain
    documents; re-downloading it is cheap and committing it would be rude.
    """

    data: Path = field(default_factory=lambda: _env_path("RFCAGENT_DATA", REPO_ROOT / "data"))

    @property
    def corpus(self) -> Path:
        return self.data / "rfc"

    @property
    def index_xml(self) -> Path:
        """The RFC Editor's metadata index. One file, ~12 MB, one HTTP request."""
        return self.corpus / "rfc-index.xml"

    @property
    def text(self) -> Path:
        """Raw RFC text files, one per document: text/rfc7234.txt."""
        return self.corpus / "text"

    @property
    def artifacts(self) -> Path:
        """Derived, rebuildable: parsed docs, chunk tables, indices, embeddings."""
        return self.data / "artifacts"

    @property
    def traces(self) -> Path:
        return self.data / "traces"

    @property
    def evals(self) -> Path:
        """Eval inputs are versioned in git; eval *outputs* are not."""
        return REPO_ROOT / "agent" / "evals"

    @property
    def ledger(self) -> Path:
        return REPO_ROOT / "experiments" / "experiments.csv"


@dataclass(frozen=True)
class ChunkSettings:
    """Chunking is a measured choice, not a default.

    Week 1 compares `section` against `window` on the same question set. Whichever
    wins, wins on recall@5 and gets a row in the ledger. See agent/docs/eval_report.md.
    """

    strategy: str = "section"  # "section" | "window"
    target_tokens: int = 400
    overlap_tokens: int = 64
    min_tokens: int = 32  # below this, a chunk is a heading with no body — drop it


@dataclass(frozen=True)
class RetrievalSettings:
    # BM25. Standard Robertson/Sparck-Jones parameters; k1 controls term-frequency
    # saturation, b controls length normalization. Both are swept in week 1 rather
    # than accepted, because the corpus has an unusual length distribution.
    bm25_k1: float = 1.2
    bm25_b: float = 0.75

    # Dense. bge-small is 384-dim, 33M params — small enough to be comfortable on
    # CPU and trivial on a 6 GB GPU, which is what makes the whole corpus
    # re-indexable in minutes rather than being a thing you do once and never
    # revisit. Cheap re-indexing is what makes the week-1 chunking ablation
    # affordable, so this choice is about iteration speed, not model quality.
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    # bge models are trained with an asymmetric query prefix. Omitting it costs
    # real recall, and it is the kind of silent failure this repo exists to catch.
    query_prefix: str = "Represent this sentence for searching relevant passages: "

    # Where the encoders run. "auto" resolves to cuda when torch reports a usable
    # GPU and cpu otherwise, so the same checkout runs on this laptop and in CI
    # (which has no GPU) with no edit. Override with RFCAGENT_DEVICE=cpu to force
    # the CPU path — worth doing at least once, because a GPU-only result is one
    # you cannot reproduce anywhere else.
    device: str = os.environ.get("RFCAGENT_DEVICE", "auto")
    # Batch size is device-dependent: bigger batches are free throughput on a GPU
    # and just memory pressure on a CPU. 6 GB of VRAM shared with a desktop
    # compositor is not 6 GB of headroom, so this is deliberately not the largest
    # batch that fits — an OOM three hours into an index build is a bad trade for
    # a few percent of throughput.
    embed_batch_size_cuda: int = 256
    embed_batch_size_cpu: int = 64

    # Reciprocal rank fusion. k=60 is the value from the original Cormack et al.
    # paper; it is swept, not assumed.
    rrf_k: int = 60

    # Reranking. Applied to the top `rerank_candidates` of the fused list.
    #
    # This is the setting the GPU actually changes. A cross-encoder scores every
    # (query, chunk) pair through a full forward pass — it cannot precompute the
    # way a bi-encoder does — so reranking 50 candidates is 50 forward passes per
    # query, and on CPU that is the dominant term in p95 latency. On a GPU the
    # candidate budget becomes a quality knob instead of a latency ceiling. Week 1
    # measures the recall/latency curve on both, because "reranking is too slow" is
    # a claim about hardware, not about reranking.
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 50
    rerank_batch_size_cuda: int = 128
    rerank_batch_size_cpu: int = 32
    rerank_enabled: bool = True

    top_k: int = 5

    def embed_batch_size(self, device: str | None = None) -> int:
        device = device or resolve_device(self.device)
        return self.embed_batch_size_cuda if device == "cuda" else self.embed_batch_size_cpu

    def rerank_batch_size(self, device: str | None = None) -> int:
        device = device or resolve_device(self.device)
        return self.rerank_batch_size_cuda if device == "cuda" else self.rerank_batch_size_cpu


@dataclass(frozen=True)
class AgentSettings:
    provider: str = os.environ.get("RFCAGENT_PROVIDER", "ollama")
    model: str = os.environ.get("RFCAGENT_MODEL", "qwen2.5:7b")
    max_turns: int = 8
    max_tokens: int = 4096
    temperature: float = 0.0

    # Below this groundedness score the agent must refuse or escalate rather than
    # answer. This threshold is set from the eval curve, not chosen by feel — see
    # the abstention section of the eval report.
    groundedness_floor: float = 0.6


@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)
    chunk: ChunkSettings = field(default_factory=ChunkSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    seed: int = SEED

    # Settings that change *where* work runs, not *what* comes out of it. These are
    # excluded from the fingerprint on purpose: recall@5 on a given chunking and
    # ranking configuration is the same number whether the encoder ran on a GPU or
    # a CPU, and letting the device into the hash would fragment the ledger into
    # two families of runs that are not actually comparing anything different.
    #
    # They are not free to ignore, though — they move latency, which *is* a
    # reported metric. So they get their own `device` column in the ledger rather
    # than being folded into `config_hash`. Quality is keyed by the hash; latency
    # is keyed by the hash *and* the device.
    _RUNTIME_ONLY_KEYS = (
        "device",
        "embed_batch_size_cuda",
        "embed_batch_size_cpu",
        "rerank_batch_size_cuda",
        "rerank_batch_size_cpu",
    )

    def fingerprint(self) -> str:
        """A short stable hash of everything that can change a reported number.

        Paths are excluded: where the corpus sits on disk does not change recall@5.
        Device and batch sizes are excluded for the reason above — see
        `_RUNTIME_ONLY_KEYS`.
        """
        retrieval = {
            k: v for k, v in asdict(self.retrieval).items() if k not in self._RUNTIME_ONLY_KEYS
        }
        payload = {
            "chunk": asdict(self.chunk),
            "retrieval": retrieval,
            "agent": asdict(self.agent),
            "seed": self.seed,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    @property
    def device(self) -> str:
        """The concrete device the encoders will run on."""
        return resolve_device(self.retrieval.device)


def set_seed(seed: int = SEED, *, deterministic: bool = True) -> None:
    """Seed every RNG this package can reach.

    `torch.manual_seed` alone is necessary and not sufficient — python's `random`
    drives the eval sampling and numpy drives the split. Missing one of the three
    is a reproducibility bug that only shows up as a number that won't replicate.

    **A seed is not enough on a GPU.** cuDNN picks convolution algorithms by
    benchmarking them at runtime, and several CUDA reductions accumulate in
    nondeterministic order, so two identically seeded runs on the same GPU can
    differ in the last few decimal places. That is usually invisible and
    occasionally flips the order of two near-tied retrieval hits, which is exactly
    the kind of drift that reads as a regression and isn't. `deterministic=True`
    pins cuDNN to reproducible algorithms; it costs some throughput, which is a
    trade worth making for a run whose number goes in the ledger. Pass
    `deterministic=False` for throughput-bound work whose output is not a reported
    metric — a one-off index build, for instance.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass


settings = Settings()
