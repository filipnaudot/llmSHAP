from llmSHAP.types import Any


class Attribution:
    """Represents an attribution result and its associated output."""

    def __init__(self, attribution: Any, output: Any) -> None:
        """
        Initialize an Attribution instance.

        Args:
            attribution: The (normalized) result/attribution data.
            output: The generated output associated with the attribution.
        """
        self._attribution = attribution
        self._output = output

    @property
    def attribution(self) -> Any:
        """Return the attribution result."""
        return self._attribution

    @property
    def output(self) -> Any:
        """Return the output data."""
        return self._output

    def render(self) -> str:
        RESET="\033[0m"
        FG="\033[38;5;0m"
        BG=lambda s:(lambda s: f"\033[48;5;{196+7*round((1-s)*4)}m" if s>=0 else f"\033[48;5;{16+42*round((1+s)*4)+5}m")(max(-1,min(1,s)))
        return " ".join(f"{BG(d.get('score',0))}{FG} {d.get('value','')} {RESET}" for d in self._attribution.values())