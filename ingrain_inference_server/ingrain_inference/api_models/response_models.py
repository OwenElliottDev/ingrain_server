from ingrain_inference.api_models.camel_model import CamelModel


class EmbeddingResponse(CamelModel):
    text_embeddings: list[list[float]] | None = None
    image_embeddings: list[list[float]] | None = None
    processing_time_ms: float


class TextEmbeddingResponse(CamelModel):
    embeddings: list[list[float]]
    processing_time_ms: float


class ImageEmbeddingResponse(CamelModel):
    embeddings: list[list[float]]
    processing_time_ms: float


class ImageClassificationResponse(CamelModel):
    probabilities: list[list[float]]
    processing_time_ms: float


class GenericMessageResponse(CamelModel):
    message: str


class InferenceStats(CamelModel):
    count: str | None = None
    ns: str | None = None


class BatchStats(CamelModel):
    batch_size: str
    compute_input: InferenceStats
    compute_infer: InferenceStats
    compute_output: InferenceStats


class ModelStats(CamelModel):
    name: str
    version: str
    last_inference: str | None = None
    inference_count: str | None = None
    execution_count: str | None = None
    inference_stats: dict[str, InferenceStats | None]
    batch_stats: list[BatchStats] | None = None


class MetricsResponse(CamelModel):
    model_stats: list[ModelStats]
