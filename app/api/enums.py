
from enum import Enum

class SocketMessageType(str, Enum):
    """
    Enumeration of message types used for socket communication
    in workflow upload and execution, and tool execution processes.
    """

    # workflow upload
    
    TOOL_INSTALL = "TOOL_INSTALL"
    UPLOAD_WORKFLOW = "UPLOAD_WORKFLOW"
    UPLOAD_FAILURE = "UPLOAD_FAILURE"
    UPLOAD_COMPLETE = "UPLOAD_COMPLETE"
    
    # Workflow Invocation
    
    INVOCATION_FAILURE = "INVOCATION_FAILURE"
    INVOCATION_STEP_UPDATE = "INVOCATION_STEP_UPDATE"
    INVOCATION_COMPLETE = "INVOCATION_COMPLETE"

    # Workflow execution
    
    WORKFLOW_EXECUTE = "WORKFLOW_EXECUTE"
    WORKFLOW_FAILURE = "WORKFLOW_FAILURE"

    # Tool Execution
    
    JOB_UPDATE = "JOB_UPDATE"
    JOB_COMPLETE = "JOB_COMPLETE"
    JOB_FAILURE = "JOB_FAILURE"
    TOOL_EXECUTE = "TOOL_EXECUTE"
    
    # Output Dataset Indexing
    
    INDEX_UPDATE = "INDEX_UPDATE"
    INDEX_START =  "INDEX_START"
    INDEX_FINISH = "INDEX_FINISH"

class SocketMessageEvent(str, Enum):
    """Enumeration of possible socket message event types for workflow and tool execution."""

    # ping
    ping = "ping"
    
    # Workflow Execution
    workflow_execute = "workflow_execute"
    
    # Workflow Execution
    workflow_upload = "workflow_upload"
    
    # Tool Execution
    tool_execute = "tool_execute"

    # Output Indexing
    output_index = "output_index"
    
    
class CollectionNames(Enum):
    """ Enumerations for mongo collection names. """
    
    INVOCATION_LISTS = "InvocationList"
    INVOCATION_IDS = "InvocationIDs"
    INVOCATION_RESULTS = "InvocationResults"
    DELETED_INVOCATIONS = "DeletedInvocation"
    INVOCATION_STATES = "InvocationStates"