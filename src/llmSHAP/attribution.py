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
        """
        Render the attribution in a heatmap format.

        TODO: Implement the rendering.
        """
        raise NotImplementedError("The render() method is not implemented yet.")