from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union

class OutputDataset(BaseModel):
    """Schema representing a single output dataset from a workflow."""
    id: str
    name: str
    file_ext: str
    visible: bool
    peek: str
    data_type: str

class CollectionOutputDataset(BaseModel):
    """Schema representing an element output dataset from a collection dataset."""
    id: str
    name: str
    visible: bool
    collection_type: str
    elements: List[Dict[str,str]]


class WorkflowExecutionResponse(BaseModel):
    """Response schema after a workflow has been executed and tracked."""
    invocation_id: str
    history_id: str
    report: Dict[str, Any] = None
    final_outputs: List[Union[OutputDataset, CollectionOutputDataset]]
    intermediate_outputs: List[Union[OutputDataset, CollectionOutputDataset]]

class WorkflowDetails(BaseModel):
    """"Schema representing the details of a single workflow"""
    id: str
    tags: Optional[List[str] ]= None
    create_time: str
    annotations: Optional[str] = None
    published: bool
    license: str
    galaxy_url: str
    creator: List[Dict]
    steps: Dict[str, Dict[str, Any]]
    inputs: Dict[str,Dict[str, Any]]