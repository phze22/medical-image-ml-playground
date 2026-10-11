
import os
from typing import Any

from databricks.sdk import WorkspaceClient


def image_to_record(image: Any) -> dict[str, Any]:
    """Convert the tensor into the endpoint's expected input format.

    Implement this according to the deployed model's input schema.
    """
    raise NotImplementedError(
        "Configure the request format for your Databricks model."
    )


def predict(record: dict[str, Any]) -> Any:
    """Call a Databricks Model Serving endpoint."""
    client = WorkspaceClient()

    return client.serving_endpoints.query(
        name=os.environ["DATABRICKS_ENDPOINT"],
        dataframe_records=[record],
    )


def extract_label(response: Any) -> str:
    """Extract the text label from the endpoint response."""
    raise NotImplementedError(
        "Configure this function using an example endpoint response."
    )
