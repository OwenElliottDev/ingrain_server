from ingrain_models.api_models.camel_model import CamelModel


class LoadedModelData(CamelModel):
    name: str
    library: str


class LoadedModelResponse(CamelModel):
    models: list[LoadedModelData]


class RepositoryModel(CamelModel):
    name: str
    library: str
    state: str | None


class RepositoryModelResponse(CamelModel):
    models: list[RepositoryModel]


class GenericMessageResponse(CamelModel):
    message: str


class ModelEmbeddingDimsResponse(CamelModel):
    embedding_size: int


class ModelClassificationLabelsResponse(CamelModel):
    labels: list[str]
