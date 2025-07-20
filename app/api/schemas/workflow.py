from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class OutputDataset(BaseModel):
    """Schema representing a single output dataset from a workflow."""
    id: str
    name: str
    file_ext: str
    visible: bool

class WorkflowExecutionResponse(BaseModel):
    """Response schema after a workflow has been executed and tracked."""
    invocation_id: str
    history_id: str
    report: Optional[Dict[str, Any]] = None
    final_outputs: List[OutputDataset]
    intermediate_outputs: List[OutputDataset]