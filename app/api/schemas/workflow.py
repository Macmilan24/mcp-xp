from pydantic import BaseModel, Field
from typing import Annotated, List, Dict, Any, Optional, Union, Literal

class OutputDataset(BaseModel):
    """Schema representing a single output dataset from a workflow."""
    type: Literal["dataset"] = "dataset"
    id: str
    name: str
    file_ext: Optional[str] = None
    visible: bool
    file_path: Optional[str] = None
    peek: Optional[str] = None
    data_type: str
    is_intermediate: Optional[bool] = None

class CollectionOutputDataset(BaseModel):
    """Schema representing an element output dataset from a collection dataset."""
    type: Literal["collection"] = "collection"
    id: str
    name: str
    visible: bool
    collection_type: str
    elements: List[Dict[str, Any]]
    is_intermediate: Optional[bool] = None

class WorkflowListItem(BaseModel):
    """Schema representing a single workflow in a list."""
    id: str
    name: str
    description: Optional[str] = None
    tags: Optional[List[str]] = []

class WorkflowList(BaseModel):
    """Schema representing a list of workflows with basic information."""
    workflows: List[WorkflowListItem]
    
class WorkflowExecutionResponse(BaseModel):
    """Response schema after a workflow has been executed and tracked."""
    invocation_id: str
    history_id: str
    report: Dict[str, Any] = None
    result: List[Annotated[
                                Union[OutputDataset, CollectionOutputDataset],
                                Field(discriminator="type")
                            ]] = []

class WorkflowDetails(BaseModel):
    """"Schema representing the details of a single workflow"""
    id: str
    tags: Optional[List[str]] = []
    create_time: str
    annotations: Optional[str] = None
    published: bool
    license: str = ""
    galaxy_url: str
    creator: List[Dict] = []
    steps: Dict[str, Dict[str, Any]]
    inputs: Dict[str,Dict[str, Any]]