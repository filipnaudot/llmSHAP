class Generation:
    def __init__(self, output: str, score: int | None = None):
        self.output: str = output
        self.score: int | None = score