"""The verdict, and how it prints."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Finding", "Report"]

LEAK = "leak"
BYPASS = "bypass"
OK = "ok"


@dataclass(frozen=True)
class Finding:
    """One (feature, source) pair, and what the perturbation did to it."""

    feature: str
    source: str
    kind: str
    declared: bool
    moved: bool
    max_abs_change: float

    def __str__(self) -> str:
        if self.kind == LEAK:
            return (
                f"{self.feature} moved when {self.source} was perturbed, and does not "
                f"declare it (max change {self.max_abs_change:.3g})"
            )
        if self.kind == BYPASS:
            return (
                f"{self.feature} declares {self.source} but did not move when its clock "
                f"did -- it reads the source without consulting availability, or the "
                f"declaration is stale"
            )
        return f"{self.feature} / {self.source}: as declared"


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def leaks(self) -> list[Finding]:
        """Undeclared dependencies. These are the bugs."""
        return [f for f in self.findings if f.kind == LEAK]

    @property
    def bypassed(self) -> list[Finding]:
        """Declared dependencies that did not respond to their own clock.

        Either the declaration is stale, or the feature reaches the source by a
        path that ignores its availability column -- which is how look-ahead
        usually gets in. Not a failure on its own; worth reading.
        """
        return [f for f in self.findings if f.kind == BYPASS]

    @property
    def clean(self) -> bool:
        return not self.leaks

    def raise_for_leaks(self) -> None:
        """For use in a test suite: turn a leak into a failure."""
        if self.leaks:
            listed = "\n".join(f"  - {f}" for f in self.leaks)
            raise AssertionError(f"{len(self.leaks)} undeclared dependencies:\n{listed}")

    def __str__(self) -> str:
        width = max((len(f) for f in self.features), default=7)
        lines = [
            f"{len(self.features)} features x {len(self.sources)} sources",
            "",
            f"{'feature':<{width}}  " + "  ".join(f"{s:>12}" for s in self.sources),
        ]
        by_pair = {(f.feature, f.source): f for f in self.findings}
        for feature in self.features:
            cells = []
            for source in self.sources:
                found = by_pair.get((feature, source))
                if found is None:
                    cells.append(f"{'-':>12}")
                elif found.kind == LEAK:
                    cells.append(f"{'LEAK':>12}")
                elif found.kind == BYPASS:
                    cells.append(f"{'bypass?':>12}")
                elif found.declared:
                    cells.append(f"{'reads it':>12}")
                else:
                    cells.append(f"{'exactly 0':>12}")
            lines.append(f"{feature:<{width}}  " + "  ".join(cells))

        lines.append("")
        if self.leaks:
            lines.append(f"{len(self.leaks)} undeclared dependencies:")
            lines += [f"  - {f}" for f in self.leaks]
        else:
            lines.append("No undeclared dependencies.")
        if self.bypassed:
            lines.append("")
            lines.append("Declared, but did not respond to the source's clock:")
            lines += [f"  - {f}" for f in self.bypassed]
        if self.notes:
            lines.append("")
            lines += self.notes
        return "\n".join(lines)
