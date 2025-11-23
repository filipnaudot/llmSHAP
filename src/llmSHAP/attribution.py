from llmSHAP.types import ResultMapping


class Attribution:
    """Represents an attribution result and its associated output."""

    def __init__(self, attribution: ResultMapping,
                 output: str,
                 baseline: float,
                 grand_coalition_value: float) -> None:
        """
        Initialize an Attribution instance.

        Args:
            attribution: The (normalized) result/attribution data.
            output: The generated output associated with the attribution.
        """
        self._attribution = attribution
        self._output = output
        self._empty_baseline = baseline
        self._grand_coalition_value = grand_coalition_value

    @property
    def attribution(self) -> ResultMapping:
        """Return the attribution result."""
        return self._attribution

    @property
    def output(self) -> str:
        """Return the output data."""
        return self._output
    
    @property
    def empty_baseline(self) -> float:
        """Return the empty baseline value (the no player pay-off)."""
        return self._empty_baseline
    
    @property
    def grand_coalition_value(self) -> float:
        """Return the value of the grand coalition."""
        return self._grand_coalition_value

    def render(self, abs_values: bool = False) -> str:
        RESET="\033[0m"
        FG="\033[38;5;0m"
        BG=lambda s:(lambda s: f"\033[48;5;{196+7*round((1-s)*4)}m" if s>=0 else f"\033[48;5;{16+42*round((1+s)*4)+5}m")(max(-1,min(1,s)))
        return " ".join(
            f"{BG(abs(item.get('score', 0)) if abs_values else item.get('score', 0))}"
            f"{FG} {item.get('value', '')} {RESET}"
            for item in self._attribution.values()
        )