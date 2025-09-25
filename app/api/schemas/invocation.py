from pydantic import BaseModel, Field
from typing import List, Dict, Annotated, Union, Any, Literal, Optional
from app.api.schemas.workflow import WorkflowExecutionResponse, OutputDataset, CollectionOutputDataset

from app.api.schemas.workflow import WorkflowListItem
class InvocationListItem(BaseModel):
    """Schema representing the details of a single invocation information."""
    id: str
    workflow_name: str
    workflow_id: str
    history_id: str
    state: Literal["Pending", "Failed", "Complete"]
    create_time: str
    update_time: str

class InvocationList(BaseModel):
    invocations: List[InvocationListItem]

class InputParameter(BaseModel):
    type: Literal["parameter"]
    value: Any


class InvocationResult(BaseModel):
    invocation_id: str
    state: Literal["Pending", "Failed", "Complete"]
    history_id: str
    create_time: str
    update_time: str
    inputs:  Dict[str, Annotated[
                            Union[CollectionOutputDataset, OutputDataset, InputParameter],
                            Field(discriminator="type")
                        ]
                    ]
    result: List[Annotated[
                                Union[OutputDataset, CollectionOutputDataset],
                                Field(discriminator="type")
                            ]] = []
    workflow:WorkflowListItem
    report : Optional[Dict[str, Any]] = None