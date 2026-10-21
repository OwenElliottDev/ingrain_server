from ingrain_inference.api_models.camel_model import CamelModel


class EmbeddingRequest(CamelModel):
    name: str
    text: str | list[str] | None = None
    image: str | list[str] | None = None
    normalize: bool | None = True
    n_dims: int | None = None
    image_download_headers: dict | None = None


class TextEmbeddingRequest(CamelModel):
    name: str
    text: str | list[str]
    normalize: bool | None = True
    n_dims: int | None = None


class ImageEmbeddingRequest(CamelModel):
    name: str
    image: str | list[str]
    normalize: bool | None = True
    n_dims: int | None = None
    image_download_headers: dict | None = None


class ImageClassificationRequest(CamelModel):
    name: str
    image: str | list[str]
    image_download_headers: dict | None = None
