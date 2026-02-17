import base64
import builtins
import pytest

from llmSHAP.image import Image


def test_str_prefers_path():
    image = Image(image_path="path/to/image.png")
    assert str(image) == "IMAGE: path/to/image.png"

def test_str_falls_back_to_url():
    image = Image(url="https://example.com/image.png")
    assert str(image) == "IMAGE: https://example.com/image.png"

def test_str_empty_when_missing_both():
    image = Image()
    assert str(image) == ""

def test_encoded_image_and_data_url(tmp_path):
    content = b"fake-image-bytes"
    image_path = tmp_path / "img.png"
    image_path.write_bytes(content)
    image = Image(image_path=str(image_path))
    encoded = image.encoded_image()
    assert encoded == base64.b64encode(content).decode("utf-8")
    assert image.data_url("image/png") == f"data:image/png;base64,{encoded}"

def test_encoded_image_requires_path():
    image = Image()
    with pytest.raises(ValueError):
        image.encoded_image()

def test_data_url_requires_mime():
    image = Image(image_path="path/to/image.png")
    with pytest.raises(ValueError):
        image.data_url("")

def test_encoded_image_is_cached_for_same_file(tmp_path, monkeypatch):
    content = b"cache-me"
    image_path = tmp_path / "img.png"
    image_path.write_bytes(content)
    image = Image(image_path=str(image_path))

    Image._encoded_from_path.cache_clear()
    open_calls = {"count": 0}
    real_open = builtins.open

    def counting_open(*args, **kwargs):
        if args and args[0] == str(image_path): open_calls["count"] += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", counting_open)
    encoded_once = image.encoded_image()
    encoded_twice = image.encoded_image()
    data_url = image.data_url("image/png")

    assert encoded_once == encoded_twice
    assert data_url == f"data:image/png;base64,{encoded_once}"
    assert open_calls["count"] == 1
