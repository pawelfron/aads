"""Train one registered model on the PubMed citation graph.

The scientific-field label of each paper is a node class. Run this module
once per model and point TensorBoard at the shared log directory to compare
approaches:

    python -m aads.train --model <name>
    tensorboard --logdir runs
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import random
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
import torch
import torch_geometric
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from tqdm import tqdm

from aads.data import CLASS_NAMES, PUBMED_NAME, SPLITS, load_pubmed, split_sizes
from aads.early_stopping import EarlyStopping
from aads.models import NodeClassifier, available_models, build_model

logger = logging.getLogger(__name__)

Monitor = Literal["val_loss", "val_accuracy"]
MONITOR_MODES: dict[Monitor, str] = {
    "val_loss": "min",
    "val_accuracy": "max",
}
SHARED_MODEL_ARGUMENTS = (
    "in_channels",
    "hidden_channels",
    "out_channels",
    "num_layers",
    "dropout",
)


class SplitMetrics(TypedDict):
    loss: float
    accuracy: float


class EpochRecord(TypedDict):
    epoch: int
    step_loss: float
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float


class MetricsRecord(TypedDict):
    monitor: str
    best: EpochRecord | None
    last: EpochRecord | None
    test: SplitMetrics | None
    epochs_completed: int
    stop_reason: str | None


@dataclass(frozen=True)
class TrainSettings:
    epochs: int
    patience: int
    min_delta: float
    monitor: Monitor
    learning_rate: float
    weight_decay: float
    device: torch.device
    run_dir: Path


def fit(
    model: NodeClassifier,
    data: Data,
    settings: TrainSettings,
    config: dict[str, Any],
) -> MetricsRecord:
    """Train ``model`` on one graph and write logs and checkpoints.

    Checkpoint selection uses the validation split. When training finishes,
    the best weights are restored and the test split is scored once.
    ``data`` and ``model`` are moved onto ``settings.device``.
    """
    settings.run_dir.mkdir(parents=True)
    _write_json(settings.run_dir / "config.json", config)

    model.to(settings.device)
    data = data.to(settings.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    stopper = EarlyStopping(
        patience=settings.patience,
        min_delta=settings.min_delta,
        mode=MONITOR_MODES[settings.monitor],
    )
    metrics: MetricsRecord = {
        "monitor": settings.monitor,
        "best": None,
        "last": None,
        "test": None,
        "epochs_completed": 0,
        "stop_reason": None,
    }
    best_path = settings.run_dir / "best.pt"
    last_path = settings.run_dir / "last.pt"
    writer = SummaryWriter(log_dir=settings.run_dir)

    try:
        writer.add_text("config", "```json\n" + json.dumps(config, indent=2) + "\n```")
        try:
            stop_reason = "max_epochs"
            failure: str | None = None
            with tqdm(total=settings.epochs, desc="train", unit="epoch") as progress:
                for epoch in range(1, settings.epochs + 1):
                    step_loss = _train_step(model, data, optimizer)
                    if not math.isfinite(step_loss):
                        stop_reason = "non_finite_loss"
                        failure = f"Training loss is not finite at epoch {epoch}."
                        break
                    scores = _evaluate_splits(model, data, ("train", "val"))
                    if not _scores_are_finite(scores):
                        stop_reason = "non_finite_loss"
                        failure = f"Evaluation metrics are not finite at epoch {epoch}."
                        break

                    record = _epoch_record(epoch, step_loss, scores["train"], scores["val"])
                    metrics["last"] = record
                    metrics["epochs_completed"] = epoch
                    _save_weights(model, last_path)
                    observed = _monitored_value(record, settings.monitor)
                    if stopper.update(observed, epoch):
                        metrics["best"] = record
                        _save_weights(model, best_path)
                    _log_epoch(writer, record)
                    _write_json(settings.run_dir / "metrics.json", metrics)
                    progress.update(1)
                    progress.set_postfix(
                        step=f"{step_loss:.4f}",
                        val_loss=f"{scores['val']['loss']:.4f}",
                        val_acc=f"{scores['val']['accuracy']:.4f}",
                    )
                    if stopper.should_stop:
                        stop_reason = "early_stopping"
                        break

            metrics["stop_reason"] = stop_reason
            if failure is not None:
                logger.error(failure)
            elif stop_reason == "early_stopping":
                best_record = metrics["best"]
                best_value = stopper.best
                if best_record is None or best_value is None:
                    raise RuntimeError("Early stopping ran before a best checkpoint was recorded.")
                logger.info(
                    "Early stopping at epoch %d. Best %s is %.4f at epoch %d.",
                    metrics["epochs_completed"],
                    settings.monitor,
                    best_value,
                    best_record["epoch"],
                )
            _evaluate_test(model, data, settings.device, best_path, metrics, writer)
            _write_json(settings.run_dir / "metrics.json", metrics)
            if metrics["test"] is not None and metrics["best"] is not None:
                _log_hparams(writer, config, metrics)
        except KeyboardInterrupt:
            metrics["stop_reason"] = "interrupted"
            _write_json(settings.run_dir / "metrics.json", metrics)
            logger.info("Training interrupted. Checkpoints are in %s.", settings.run_dir)
            raise
    finally:
        writer.close()

    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.list_models:
        names = available_models()
        print("\n".join(names) if names else "No models are registered.")
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    _require_registered_model(args.model)
    _reject_overlapping_model_kwargs(args.model_kwargs)
    _set_seed(args.seed)
    device = _resolve_device(args.device)

    dataset, data = load_pubmed(
        args.data_dir,
        split=args.split,
        num_train_per_class=args.num_train_per_class,
        num_val=args.num_val,
        num_test=args.num_test,
    )
    if dataset.num_classes != len(CLASS_NAMES):
        raise RuntimeError(
            f"PubMed exposes {dataset.num_classes} classes, "
            f"but {len(CLASS_NAMES)} class names are defined."
        )
    sizes = split_sizes(data)
    empty = [name for name, count in sizes.items() if count == 0]
    if empty:
        raise SystemExit(f"The PubMed split has no nodes in {', '.join(empty)}.")
    if data.num_nodes is None:
        raise RuntimeError("PubMed graph does not know how many nodes it has.")

    try:
        model = build_model(
            args.model,
            in_channels=dataset.num_features,
            hidden_channels=args.hidden_channels,
            out_channels=dataset.num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
            **args.model_kwargs,
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    run_dir = _resolve_run_dir(args.log_dir, args.model, args.run_name)
    recorded_argv = list(sys.argv[1:] if argv is None else argv)
    config = _run_config(
        args=args,
        argv=recorded_argv,
        device=device,
        run_dir=run_dir,
        num_nodes=int(data.num_nodes),
        num_edges=int(data.num_edges),
        num_features=int(dataset.num_features),
        num_classes=int(dataset.num_classes),
        split_sizes_by_name=sizes,
    )
    settings = TrainSettings(
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        monitor=args.monitor,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        run_dir=run_dir,
    )

    logger.info(
        "Loaded %s (%s split): %d nodes, %d edges, %d features, %d classes.",
        PUBMED_NAME,
        args.split,
        data.num_nodes,
        data.num_edges,
        dataset.num_features,
        dataset.num_classes,
    )
    logger.info(
        "Split sizes: train=%d val=%d test=%d.",
        sizes["train"],
        sizes["val"],
        sizes["test"],
    )
    logger.info(
        "Classes: %s.",
        ", ".join(f"{index}={name}" for index, name in enumerate(CLASS_NAMES)),
    )
    logger.info("Training %s on %s. Run directory: %s.", args.model, device, run_dir)

    try:
        metrics = fit(model, data, settings, config)
    except KeyboardInterrupt:
        raise SystemExit(130) from None

    best = metrics["best"]
    test = metrics["test"]
    if best is None or test is None:
        logger.error("Training finished without a saved checkpoint.")
        raise SystemExit(1)
    logger.info(
        "Finished after %d epoch(s) (%s).",
        metrics["epochs_completed"],
        metrics["stop_reason"],
    )
    logger.info(
        "Best epoch %d: val loss %.4f, val accuracy %.4f.",
        best["epoch"],
        best["val_loss"],
        best["val_accuracy"],
    )
    logger.info("Test loss %.4f, test accuracy %.4f.", test["loss"], test["accuracy"])
    logger.info("Saved best model to %s.", run_dir / "best.pt")
    logger.info("Saved last model to %s.", run_dir / "last.pt")
    logger.info("To compare runs: tensorboard --logdir %s", args.log_dir)


def _train_step(model: NodeClassifier, data: Data, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(data.x, data.edge_index)
    loss = nn.functional.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    if not torch.isfinite(loss):
        return float("nan")
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def _evaluate_splits(
    model: NodeClassifier,
    data: Data,
    splits: tuple[str, ...],
) -> dict[str, SplitMetrics]:
    model.eval()
    logits = model(data.x, data.edge_index)
    scores: dict[str, SplitMetrics] = {}
    for split in splits:
        mask = data[f"{split}_mask"]
        split_logits = logits[mask]
        split_labels = data.y[mask]
        loss = nn.functional.cross_entropy(split_logits, split_labels)
        scores[split] = {
            "loss": float(loss.item()),
            "accuracy": _accuracy(split_logits, split_labels),
        }
    return scores


def _evaluate_test(
    model: NodeClassifier,
    data: Data,
    device: torch.device,
    best_path: Path,
    metrics: MetricsRecord,
    writer: SummaryWriter,
) -> None:
    best = metrics["best"]
    if best is None:
        logger.error("No checkpoint was saved, so the test split was not scored.")
        return
    state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    test = _evaluate_splits(model, data, ("test",))["test"]
    metrics["test"] = test
    writer.add_scalar("loss/test", test["loss"], best["epoch"])
    writer.add_scalar("accuracy/test", test["accuracy"], best["epoch"])


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    correct = int(predictions.eq(labels).sum().item())
    return correct / labels.shape[0]


def _epoch_record(
    epoch: int,
    step_loss: float,
    train: SplitMetrics,
    val: SplitMetrics,
) -> EpochRecord:
    return {
        "epoch": epoch,
        "step_loss": step_loss,
        "train_loss": train["loss"],
        "train_accuracy": train["accuracy"],
        "val_loss": val["loss"],
        "val_accuracy": val["accuracy"],
    }


def _scores_are_finite(scores: dict[str, SplitMetrics]) -> bool:
    return all(
        math.isfinite(metrics["loss"]) and math.isfinite(metrics["accuracy"])
        for metrics in scores.values()
    )


def _monitored_value(record: EpochRecord, monitor: Monitor) -> float:
    if monitor == "val_loss":
        return record["val_loss"]
    return record["val_accuracy"]


def _log_epoch(writer: SummaryWriter, record: EpochRecord) -> None:
    epoch = record["epoch"]
    writer.add_scalar("loss/step", record["step_loss"], epoch)
    writer.add_scalar("loss/train", record["train_loss"], epoch)
    writer.add_scalar("loss/val", record["val_loss"], epoch)
    writer.add_scalar("accuracy/train", record["train_accuracy"], epoch)
    writer.add_scalar("accuracy/val", record["val_accuracy"], epoch)
    writer.flush()


def _log_hparams(writer: SummaryWriter, config: dict[str, Any], metrics: MetricsRecord) -> None:
    best = metrics["best"]
    test = metrics["test"]
    if best is None or test is None:
        return
    hparams: dict[str, str | int | float | bool] = {}
    for key in (
        "model",
        "hidden_channels",
        "num_layers",
        "dropout",
        "lr",
        "weight_decay",
        "epochs",
        "patience",
        "min_delta",
        "monitor",
        "seed",
        "split",
    ):
        value = config.get(key)
        if isinstance(value, (str, int, float, bool)):
            hparams[key] = value
    model_kwargs = config.get("model_kwargs", {})
    if isinstance(model_kwargs, dict):
        for key, value in model_kwargs.items():
            if isinstance(value, (str, int, float, bool)):
                hparams[f"kw_{key}"] = value
            else:
                hparams[f"kw_{key}"] = json.dumps(value)
    writer.add_hparams(
        hparams,
        {
            "best_epoch": best["epoch"],
            "best_val_loss": best["val_loss"],
            "best_val_accuracy": best["val_accuracy"],
            "test_loss": test["loss"],
            "test_accuracy": test["accuracy"],
        },
        run_name="hparams",
    )


def _save_weights(model: nn.Module, path: Path) -> None:
    torch.save(model.state_dict(), path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _run_config(
    *,
    args: argparse.Namespace,
    argv: list[str],
    device: torch.device,
    run_dir: Path,
    num_nodes: int,
    num_edges: int,
    num_features: int,
    num_classes: int,
    split_sizes_by_name: dict[str, int],
) -> dict[str, Any]:
    return {
        "model": args.model,
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "model_kwargs": args.model_kwargs,
        "optimizer": "adam",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "monitor": args.monitor,
        "seed": args.seed,
        "split": args.split,
        "num_train_per_class": args.num_train_per_class,
        "num_val": args.num_val,
        "num_test": args.num_test,
        "data_dir": str(args.data_dir),
        "log_dir": str(args.log_dir),
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "device": str(device),
        "argv": argv,
        "dataset": {
            "name": PUBMED_NAME,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_features": num_features,
            "num_classes": num_classes,
            "class_names": list(CLASS_NAMES),
            "split_sizes": split_sizes_by_name,
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
        },
    }


def _require_registered_model(name: str) -> None:
    known = available_models()
    if name in known:
        return
    rendered = ", ".join(known) if known else "none"
    raise SystemExit(f"Unknown model {name!r}. Registered models: {rendered}.")


def _reject_overlapping_model_kwargs(model_kwargs: dict[str, Any]) -> None:
    overlap = sorted(set(SHARED_MODEL_ARGUMENTS).intersection(model_kwargs))
    if overlap:
        joined = ", ".join(overlap)
        raise SystemExit(
            f"--model-kwargs repeats shared hyperparameters: {joined}. "
            "Pass those with their own flags."
        )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(name)
    except RuntimeError as exc:
        raise SystemExit(f"Unknown device {name!r}.") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"Device {name!r} is not available.")
    return device


def _resolve_run_dir(log_dir: Path, model_name: str, run_name: str | None) -> Path:
    if run_name is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name).strip("-") or "model"
        run_name = f"{safe_model}-{stamp}"
    if run_name in {".", ".."} or "/" in run_name or "\\" in run_name:
        raise SystemExit("--run-name must be a single directory name, without path separators.")
    run_dir = log_dir / run_name
    if run_dir.exists():
        raise SystemExit(f"Run directory already exists: {run_dir}")
    return run_dir


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="aads",
        description=(
            "Train a graph neural network to classify the scientific field of "
            "PubMed papers. The model is selected by name from the model registry."
        ),
        epilog=(
            "Each invocation trains one registered model. Compare approaches with "
            "TensorBoard pointed at the log directory: tensorboard --logdir runs"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print registered model names and exit",
    )
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", help="Registered model name")
    model_group.add_argument(
        "--hidden-channels",
        type=_positive_int,
        default=16,
        help="Width of hidden layers",
    )
    model_group.add_argument(
        "--num-layers",
        type=_positive_int,
        default=2,
        help="Number of message-passing layers",
    )
    model_group.add_argument(
        "--dropout",
        type=_unit_interval,
        default=0.5,
        help="Dropout probability passed to the model",
    )
    model_group.add_argument(
        "--model-kwargs",
        type=_json_object,
        default={},
        help='Extra constructor arguments as a JSON object, for example \'{"heads": 8}\'',
    )
    optim_group = parser.add_argument_group("optimization")
    optim_group.add_argument("--lr", type=_positive_float, default=0.01, help="Adam learning rate")
    optim_group.add_argument(
        "--weight-decay",
        type=_non_negative_float,
        default=5e-4,
        help="Adam weight decay",
    )
    optim_group.add_argument(
        "--epochs",
        type=_positive_int,
        default=200,
        help="Maximum number of epochs",
    )
    stop_group = parser.add_argument_group("early stopping")
    stop_group.add_argument(
        "--patience",
        type=_positive_int,
        default=50,
        help="Epochs without improvement on --monitor before training stops",
    )
    stop_group.add_argument(
        "--min-delta",
        type=_non_negative_float,
        default=0.0,
        help="Smallest change in --monitor that counts as an improvement",
    )
    stop_group.add_argument(
        "--monitor",
        choices=tuple(MONITOR_MODES),
        default="val_loss",
        help="Validation metric used for early stopping and the best checkpoint",
    )
    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory for the downloaded PubMed files",
    )
    data_group.add_argument(
        "--split",
        choices=SPLITS,
        default="public",
        help="Planetoid split. The random split is drawn from --seed",
    )
    data_group.add_argument(
        "--num-train-per-class",
        type=_positive_int,
        default=20,
        help="Training papers per class when --split=random",
    )
    data_group.add_argument(
        "--num-val",
        type=_positive_int,
        default=500,
        help="Validation papers when --split=random",
    )
    data_group.add_argument(
        "--num-test",
        type=_positive_int,
        default=1000,
        help="Test papers when --split=random",
    )
    data_group.add_argument(
        "--seed",
        type=_non_negative_int,
        default=42,
        help="Seed for model initialization and the random split",
    )
    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--log-dir",
        type=Path,
        default=Path("runs"),
        help="Directory that stores one subdirectory per run",
    )
    output_group.add_argument(
        "--run-name",
        default=None,
        help="Directory name under --log-dir. Defaults to the model name and a timestamp",
    )
    output_group.add_argument(
        "--device",
        default="auto",
        help="cpu, cuda, cuda:N, or auto",
    )
    args = parser.parse_args(None if argv is None else list(argv))
    if not args.list_models and args.model is None:
        parser.error("--model is required")
    return args


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected an integer >= 1")
    return number


def _non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("expected an integer >= 0")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("expected a finite number > 0")
    return number


def _non_negative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("expected a finite number >= 0")
    return number


def _unit_interval(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number < 1:
        raise argparse.ArgumentTypeError("expected a float in [0, 1)")
    return number


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return parsed


if __name__ == "__main__":
    main()
