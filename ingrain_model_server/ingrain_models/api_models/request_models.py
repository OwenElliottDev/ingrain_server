from typing import Literal

from ingrain_models.api_models.camel_model import CamelModel


class LoadModelRequest(CamelModel):
    name: str
    library: Literal["open_clip", "sentence_transformers", "timm"]


class UnloadModelRequest(CamelModel):
    name: str


class DownloadCustomModelRequest(CamelModel):
    library: Literal["open_clip", "sentence_transformers", "timm"]
    name: str
    safetensors_url: str
    config_json_url: str | None = None  # sentence_transformers
    tokenizer_json_url: str | None = None  # sentence_transformers
    tokenizer_config_json_url: str | None = None  # sentence_transformers
    vocab_txt_url: str | None = None  # sentence_transformers
    special_tokens_map_json_url: str | None = None  # sentence_transformers
    pooling_config_json_url: str | None = None  # sentence_transformers
    sentence_bert_config_json_url: str | None = None  # sentence_transformers
    modules_json_url: str | None = None  # sentence_transformers
    mode: str | None = None  # open_clip
    mean: list[float] | None = None  # open_clip
    std: list[float] | None = None  # open_clip
    interpolation: str | None = None  # open_clip
    resize_mode: str | None = None  # open_clip
    num_classes: int | None = None  # timm
