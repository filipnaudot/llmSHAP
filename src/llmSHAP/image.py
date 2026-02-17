import base64
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


@dataclass(frozen=True)
class Image:
    """Image handler for local paths or remote URLs.

    Args:
        url: Remote image URL.
        image_path: Local image file path.

    Example:
        image = Image(image_path="path/to/image.jpg")
        data = {
            "image": image.data_url(mime_type="image/jpeg"),
        }
    """

    url: Optional[str] = None
    image_path: Optional[str] = None

    @staticmethod
    @lru_cache(maxsize=256)
    def _encoded_from_path(image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def encoded_image(self) -> str:
        """Return the base64-encoded contents of the local image."""
        if not self.image_path:
            raise ValueError("image_path is required to encode an image.")
        return self._encoded_from_path(self.image_path)

    def data_url(self, mime_type: str) -> str:
        """Return a data URL for the local image using the given MIME type."""
        if not mime_type:
            raise ValueError("mime_type is required to build a data URL.")
        return f"data:{mime_type};base64,{self.encoded_image()}"

    def __str__(self) -> str:
        """Return a string representation preferring path, then URL."""
        return f"IMAGE: {self.image_path}" if self.image_path else (f"IMAGE: {self.url}" if self.url else "")

    def to_string(self) -> str:
        """Return a string representation of the image."""
        return str(self)
