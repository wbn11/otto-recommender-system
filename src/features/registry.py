"""Explicit, versioned registry for the 49 M8 ranking features.

The registry is deliberately data rather than prose hidden in a training
script.  LightGBM training, streaming inference, ablation and README tables
can therefore consume the same feature names and meanings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from data.schemas import RANKER_FEATURE_SCHEMA, RANKER_FEATURE_SCHEMA_VERSION


@dataclass(frozen=True)
class Feature:
    name: str
    group: str
    meaning: str
    rationale: str
    missing_value: int | float


def _source_features(source: str, label: str) -> list[Feature]:
    return [
        Feature(
            f"from_{source}",
            "recall",
            f"Whether {label} recalled this candidate.",
            "Source agreement and source-specific reliability are strong ranking signals.",
            0,
        ),
        Feature(
            f"{source}_rank",
            "recall",
            f"One-based candidate rank inside {label}; 0 means absent.",
            "Within-source order is comparable even when raw score scales differ.",
            0,
        ),
        Feature(
            f"{source}_score",
            "recall",
            f"Raw score emitted by {label}; 0 means absent.",
            "Retains confidence information that is lost by a binary source flag.",
            0.0,
        ),
    ]


FEATURES: tuple[Feature, ...] = tuple(
    [
        Feature(
            "target_type_id",
            "base",
            "Prediction target: 1 click, 2 cart, 3 order.",
            "A unified ranker must learn different preferences for the three objectives.",
            0,
        ),
        Feature("session_length", "session", "Number of query-history events.",
                "Long histories contain more evidence and need different score calibration.", 0),
        Feature("session_click_count", "session", "Click events in query history.",
                "Describes current browsing intensity.", 0),
        Feature("session_cart_count", "session", "Cart events in query history.",
                "Signals stronger purchase intent than clicks.", 0),
        Feature("session_order_count", "session", "Order events in query history.",
                "Signals the strongest observed intent.", 0),
        Feature("session_duration_ms", "session", "Last timestamp minus first timestamp.",
                "Separates concentrated visits from long-running sessions.", 0),
        Feature("session_last_event_type", "session", "Type of the most recent event.",
                "The latest action is a compact intent signal.", 0),
        Feature("session_hour_utc", "session", "UTC hour of the most recent query event.",
                "Shopping behavior and catalog demand vary by hour.", 0),
        Feature("session_weekday_utc", "session", "UTC weekday, Monday=0.",
                "Captures weekly demand patterns.", 0),
        Feature("item_total_count", "item", "All snapshot events for this item.",
                "Global popularity is a robust prior for sparse sessions.", 0),
        Feature("item_click_count", "item", "Snapshot click count for this item.",
                "Separates broad browsing popularity from purchase intent.", 0),
        Feature("item_cart_count", "item", "Snapshot cart count for this item.",
                "Measures mid-funnel item demand.", 0),
        Feature("item_order_count", "item", "Snapshot order count for this item.",
                "Measures conversion-oriented item demand.", 0),
    ]
    + sum(
        (
            _source_features(source, label)
            for source, label in (
                ("popular", "Popular"),
                ("revisit", "Revisit"),
                ("type_covis", "Type-CoVis"),
                ("buy2buy", "Buy2Buy"),
                ("time_covis", "Time-CoVis"),
                ("dssm", "DSSM"),
            )
        ),
        [],
    )
    + [
        Feature("source_count", "recall", "Number of recall sources containing the candidate.",
                "Agreement across independent sources is a confidence signal.", 0),
        Feature("best_source_rank", "recall", "Best one-based rank across recall sources.",
                "Summarizes the strongest source position without mixing score scales.", 0),
        Feature("candidate_seen", "interaction", "Candidate occurred in query history.",
                "Repeated consumption is exceptionally strong in OTTO sessions.", 0),
        Feature("candidate_occurrence_count", "interaction",
                "Number of candidate occurrences in query history.",
                "Repeated actions distinguish persistent interest from a single view.", 0),
        Feature("candidate_last_position", "interaction",
                "One-based position of the candidate's latest history event; 0 if unseen.",
                "Keeps absolute sequential location information.", 0),
        Feature("candidate_distance_to_end", "interaction",
                "Events after the candidate's latest occurrence; -1 if unseen.",
                "A direct recency signal that is comparable across positions.", -1),
        Feature("candidate_last_event_type", "interaction",
                "Latest event type on the candidate; 0 if unseen.",
                "A recent cart/order is stronger than a recent click.", 0),
        Feature("last_item_type_covis_score", "interaction",
                "Type-CoVis score from the last history item to the candidate.",
                "Measures immediate sequential affinity.", 0.0),
        Feature("last_item_buy2buy_score", "interaction",
                "Buy2Buy score from the last history item to the candidate.",
                "Measures immediate cart/order affinity.", 0.0),
        Feature("last_item_time_covis_score", "interaction",
                "Time-CoVis score from the last history item to the candidate.",
                "Measures immediate short-time affinity.", 0.0),
        Feature("recent5_type_covis_max", "interaction",
                "Maximum Type-CoVis score from the last five history events.",
                "Finds a strong match to any recent intent.", 0.0),
        Feature("recent5_type_covis_mean", "interaction",
                "Mean Type-CoVis score over the last five history events, missing edges as 0.",
                "Measures consistency across recent intent rather than a single match.", 0.0),
        Feature("recent5_buy2buy_max", "interaction",
                "Maximum Buy2Buy score from the last five history events.",
                "Finds a strong recent purchase-oriented relation.", 0.0),
        Feature("recent5_buy2buy_mean", "interaction",
                "Mean Buy2Buy score over the last five history events, missing edges as 0.",
                "Measures sustained purchase-oriented affinity.", 0.0),
        Feature("recent5_time_covis_max", "interaction",
                "Maximum Time-CoVis score from the last five history events.",
                "Finds the strongest recent time-local relation.", 0.0),
        Feature("recent5_time_covis_mean", "interaction",
                "Mean Time-CoVis score over the last five history events, missing edges as 0.",
                "Measures sustained short-time affinity.", 0.0),
        Feature("item_recent_1d_count", "temporal",
                "Item events in the one day before the point-in-time cutoff.",
                "Captures very recent trends without validation leakage.", 0),
        Feature("item_recent_7d_count", "temporal",
                "Item events in the seven days before the point-in-time cutoff.",
                "Provides a smoother recent-demand signal.", 0),
    ]
)


FEATURE_NAMES = tuple(feature.name for feature in FEATURES)
FEATURE_GROUPS = {
    group: tuple(feature.name for feature in FEATURES if feature.group == group)
    for group in ("base", "session", "item", "recall", "interaction", "temporal")
}


def validate_registry() -> None:
    if len(FEATURES) != 49 or len(set(FEATURE_NAMES)) != len(FEATURE_NAMES):
        raise AssertionError("The M8 registry must contain 49 unique features")
    schema_feature_names = tuple(RANKER_FEATURE_SCHEMA.names[6:])
    if schema_feature_names != FEATURE_NAMES:
        raise AssertionError(
            "Feature registry and RANKER_FEATURE_SCHEMA are not in the same order"
        )


def registry_payload(enabled_groups: Iterable[str]) -> dict[str, object]:
    validate_registry()
    enabled = tuple(enabled_groups)
    unknown = set(enabled) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
    selected = [feature for feature in FEATURES if feature.group in enabled]
    return {
        "schema_version": RANKER_FEATURE_SCHEMA_VERSION,
        "feature_count": len(selected),
        "all_feature_count": len(FEATURES),
        "enabled_groups": list(enabled),
        "groups": {name: list(values) for name, values in FEATURE_GROUPS.items()},
        "features": [asdict(feature) for feature in selected],
    }


def feature_names_for_groups(enabled_groups: Iterable[str]) -> tuple[str, ...]:
    """Return canonical feature order for a whole-group selection."""

    enabled = tuple(enabled_groups)
    unknown = set(enabled) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
    enabled_set = set(enabled)
    names = tuple(feature.name for feature in FEATURES if feature.group in enabled_set)
    if not names:
        raise ValueError("At least one feature group must be enabled")
    return names


def registry_payload_for_features(feature_names: Iterable[str]) -> dict[str, object]:
    """Build model schema metadata for a canonical feature subset.

    Feature shards always retain all 49 columns.  A selected ranker may use a
    subset, so its model schema records that subset while preserving canonical
    registry order for deterministic training and inference.
    """

    validate_registry()
    requested = tuple(feature_names)
    if not requested:
        raise ValueError("At least one model feature is required")
    if len(set(requested)) != len(requested):
        raise ValueError("Model feature names must be unique")
    unknown = set(requested) - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"Unknown model features: {sorted(unknown)}")
    canonical = tuple(name for name in FEATURE_NAMES if name in set(requested))
    if requested != canonical:
        raise ValueError("Model features must follow canonical registry order")
    selected = [feature for feature in FEATURES if feature.name in set(requested)]
    enabled_groups = tuple(dict.fromkeys(feature.group for feature in selected))
    return {
        "schema_version": RANKER_FEATURE_SCHEMA_VERSION,
        "feature_count": len(selected),
        "all_feature_count": len(FEATURES),
        "enabled_groups": list(enabled_groups),
        "groups": {name: list(values) for name, values in FEATURE_GROUPS.items()},
        "features": [asdict(feature) for feature in selected],
    }


validate_registry()
