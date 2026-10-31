"""Loader, validator and prompt-builder for the retail semantic layer.

The semantic layer is the contract between the business and the LLM:
  * which tables exist and what their columns mean
  * how tables join
  * what each metric actually computes
  * which synonyms map to which metric/dimension

`prompt_context()` renders it into a compact block for the model, and
`table_names()` / `allowed_columns()` feed the SQL validator.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyYAML is required for the semantic layer. Install it with: pip install pyyaml"
    ) from exc

DEFAULT_PATH = Path(__file__).with_name("semantic_layer.yml")


class SemanticLayerError(Exception):
    """Raised when the semantic layer file is missing or structurally invalid."""


@dataclass
class Entity:
    name: str
    label: str
    columns: dict[str, dict]
    description: str = ""
    grain: str = ""
    primary_key: str | None = None

    @property
    def column_names(self) -> list[str]:
        return list(self.columns.keys())


@dataclass
class Dimension:
    id: str
    label: str
    sql: str
    type: str = "text"
    synonyms: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class Metric:
    id: str
    label: str
    sql: str
    format: str = "number"
    description: str = ""
    synonyms: list[str] = field(default_factory=list)


@dataclass
class Join:
    left: str
    right: str
    on: str
    type: str = "left"
    relationship: str = "many_to_one"


@dataclass
class SemanticLayer:
    name: str
    description: str
    default_limit: int
    entities: dict[str, Entity]
    dimensions: dict[str, Dimension]
    metrics: dict[str, Metric]
    joins: list[Join]
    examples: list[dict]
    excluded_tables: dict[str, str]
    version: int = 1
    source_path: Path | None = None

    # -- accessors ---------------------------------------------------------

    def table_names(self) -> list[str]:
        return list(self.entities.keys())

    def allowed_columns(self) -> dict[str, set[str]]:
        return {name: set(e.column_names) for name, e in self.entities.items()}

    def metric(self, metric_id: str) -> Metric | None:
        return self.metrics.get(metric_id)

    def dimension(self, dimension_id: str) -> Dimension | None:
        return self.dimensions.get(dimension_id)

    def resolve_term(self, term: str) -> tuple[str, str] | None:
        """Map a business word to ('metric'|'dimension', id) using synonyms.

        Returns None when the term is unknown. Last match wins on overlaps,
        so longer/rarer synonyms should be listed first in the YAML.
        """
        term = (term or "").strip().lower()
        if not term:
            return None

        for mid, metric in self.metrics.items():
            if term == mid.lower() or term in [s.lower() for s in metric.synonyms]:
                return ("metric", mid)

        for did, dim in self.dimensions.items():
            if term == did.lower() or term in [s.lower() for s in dim.synonyms]:
                return ("dimension", did)

        return None

    # -- prompt rendering --------------------------------------------------

    def prompt_context(self, max_examples: int = 4) -> str:
        parts: list[str] = []

        parts.append("### Tables")
        for entity in self.entities.values():
            head = f"{entity.name}"
            if entity.label and entity.label.lower() != entity.name.lower():
                head += f" ({entity.label})"
            if entity.grain:
                head += f" -- {entity.grain}"
            parts.append(head)

            for col, meta in entity.columns.items():
                desc = meta.get("description", "")
                ctype = meta.get("type", "")
                parts.append(f"  {col} {ctype}{' -- ' + desc if desc else ''}")

        parts.append("\n### Join paths (use exactly these conditions)")
        for join in self.joins:
            parts.append(f"  {join.on}")

        parts.append("\n### Metrics (use these expressions verbatim)")
        for metric in self.metrics.values():
            aliases = f" [aliases: {', '.join(metric.synonyms)}]" if metric.synonyms else ""
            parts.append(f"  {metric.id}: {metric.sql} -- {metric.label}{aliases}")

        parts.append("\n### Dimensions")
        for dim in self.dimensions.values():
            aliases = f" [aliases: {', '.join(dim.synonyms)}]" if dim.synonyms else ""
            parts.append(f"  {dim.id}: {dim.sql} -- {dim.label}{aliases}")

        if self.examples:
            parts.append("\n### Examples")
            for ex in self.examples[:max_examples]:
                sql = " ".join(str(ex.get("sql", "")).split())
                parts.append(f"Q: {ex.get('question', '').strip()}")
                parts.append(f"SQL: {sql}")

        return "\n".join(parts)

    # -- validation --------------------------------------------------------

    def validate(self) -> list[str]:
        """Structural checks. Returns a list of human-readable problems."""
        problems: list[str] = []

        if not self.entities:
            problems.append("no entities defined")

        for join in self.joins:
            for side in (join.left, join.right):
                if side not in self.entities:
                    problems.append(f"join references unknown table '{side}'")
            if "=" not in join.on:
                problems.append(f"join condition is not an equality: {join.on!r}")

        table_names = set(self.entities)

        for dim in self.dimensions.values():
            if not dim.sql.strip():
                problems.append(f"dimension '{dim.id}' has empty sql")
            elif not any(t in dim.sql for t in table_names) and "(" not in dim.sql:
                problems.append(f"dimension '{dim.id}' does not reference a known table")

        for metric in self.metrics.values():
            if not metric.sql.strip():
                problems.append(f"metric '{metric.id}' has empty sql")
            if metric.sql.count("(") != metric.sql.count(")"):
                problems.append(f"metric '{metric.id}' has unbalanced parentheses")

        # Duplicate synonyms across metrics/dimensions would make resolve_term
        # ambiguous; surface it rather than silently preferring one.
        seen: dict[str, str] = {}
        for kind, items in (("metric", self.metrics), ("dimension", self.dimensions)):
            for item_id, item in items.items():
                for syn in item.synonyms:
                    key = syn.strip().lower()
                    if not key:
                        continue
                    if key in seen:
                        problems.append(
                            f"synonym '{syn}' used by both {seen[key]} and {kind} '{item_id}'"
                        )
                    else:
                        seen[key] = f"{kind} '{item_id}'"

        return problems


def _require(data: dict, key: str, where: str) -> Any:
    if key not in data:
        raise SemanticLayerError(f"missing required key '{key}' in {where}")
    return data[key]


def parse(data: dict) -> SemanticLayer:
    if data.get("version") not in (None, 1):
        raise SemanticLayerError(f"unsupported semantic layer version: {data.get('version')}")

    entities: dict[str, Entity] = {}
    for raw in _require(data, "entities", "semantic layer"):
        name = _require(raw, "name", "entity")
        entities[name] = Entity(
            name=name,
            label=raw.get("label", name),
            columns={c: (meta or {}) for c, meta in _require(raw, "columns", f"entity '{name}'").items()},
            description=raw.get("description", ""),
            grain=raw.get("grain", ""),
            primary_key=raw.get("primary_key"),
        )

    dimensions: dict[str, Dimension] = {}
    for raw in data.get("dimensions", []) or []:
        did = _require(raw, "id", "dimension")
        dimensions[did] = Dimension(
            id=did,
            label=raw.get("label", did),
            sql=_require(raw, "sql", f"dimension '{did}'"),
            type=raw.get("type", "text"),
            synonyms=list(raw.get("synonyms") or []),
            description=raw.get("description", ""),
        )

    metrics: dict[str, Metric] = {}
    for raw in data.get("metrics", []) or []:
        mid = _require(raw, "id", "metric")
        metrics[mid] = Metric(
            id=mid,
            label=raw.get("label", mid),
            sql=_require(raw, "sql", f"metric '{mid}'"),
            format=raw.get("format", "number"),
            description=raw.get("description", ""),
            synonyms=list(raw.get("synonyms") or []),
        )

    joins: list[Join] = []
    for raw in data.get("joins", []) or []:
        joins.append(
            Join(
                left=_require(raw, "left", "join"),
                right=_require(raw, "right", "join"),
                # NOTE: the key is "condition", not "on". Unquoted `on:` is a
                # YAML 1.1 boolean and silently becomes the key True.
                on=_require(raw, "condition", "join"),
                type=raw.get("type", "left"),
                relationship=raw.get("relationship", "many_to_one"),
            )
        )

    layer = SemanticLayer(
        name=data.get("name", "retail"),
        description=data.get("description", ""),
        default_limit=int(data.get("default_limit", 500)),
        entities=entities,
        dimensions=dimensions,
        metrics=metrics,
        joins=joins,
        examples=list(data.get("examples") or []),
        excluded_tables=dict(data.get("excluded_tables") or {}),
        version=int(data.get("version", 1) or 1),
    )
    return layer


_CACHE: dict[str, SemanticLayer] = {}


def load(path: str | os.PathLike | None = None, force: bool = False) -> SemanticLayer:
    """Load and validate the semantic layer, caching by resolved path."""
    resolved = Path(path or os.environ.get("SEMANTIC_LAYER_PATH") or DEFAULT_PATH).resolve()

    key = str(resolved)
    if not force and key in _CACHE:
        return _CACHE[key]

    if not resolved.exists():
        raise SemanticLayerError(f"semantic layer not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise SemanticLayerError(f"semantic layer must be a YAML mapping: {resolved}")

    layer = parse(data)
    layer.source_path = resolved

    problems = layer.validate()
    if problems:
        raise SemanticLayerError(
            "invalid semantic layer:\n  - " + "\n  - ".join(problems)
        )

    _CACHE[key] = layer
    return layer


if __name__ == "__main__":
    layer = load()
    print(f"Loaded '{layer.name}' from {layer.source_path}")
    print(f"  entities:   {len(layer.entities)}")
    print(f"  dimensions: {len(layer.dimensions)}")
    print(f"  metrics:    {len(layer.metrics)}")
    print(f"  joins:      {len(layer.joins)}")
    print(f"  examples:   {len(layer.examples)}")
    print(f"  prompt ctx: {len(layer.prompt_context())} chars")
