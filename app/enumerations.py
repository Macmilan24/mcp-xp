from enum import IntEnum, Enum

class NumericLimits(IntEnum):
    """Centralizing numerical limits for workflows and invocations. """
    
    INVOCATION_LIMIT = 100
    RATE_RESET_TIME= 60
    BATCH_SIZE = 10
    SEMAPHORE_LIMIT = 15
    TIMEOUT = 10
    SHORT_SLEEP = 2
    BACKGROUND_INTERVAL = 600
    LONG_SLEEP = 60
    WARM_TIMESTAMP = 3600
    WARM_CHECK = 180
    BACKGROUND_INVOCATION_TRACK = 86400
    TOOL_EXECUTION_POLL = 5
    BATCH_LIMIT = 2


class TTLiveConfig(IntEnum):
    """ Centralizing TTL for invocation and worklows. """
    
    WORKFLOW_CACHE = 600
    INVOCATION_WORKFLOW_MAPPING = 120
    INVOCATION_LIST = 10
    RAW_INVOCATION_LIST = 10
    DUPLICATE_CHECK = 3
    INVOCACTION_RESULT = 259200  # 3 day cache for invocation result.
    
    
class InvocationTracking(IntEnum):
    """ Invocation tracker time limits. """
    
    BASE_NO_PROGRESS = 12 * 3600      # 12h
    STALLED_THRESHOLD = 48 * 3600     # 3 days
    HARD_CAP = 7 * 24 * 3600          # 7 days

    POLL_FAST = 5 * 60 # 5 min
    POLL_MEDIUM = 10 * 60 # 10 min
    POLL_SLOW = 15 * 60 # 15 min
    POLL_MAX = 60 * 60 # 1 hour


    
    
class JobState(str, Enum):
    """ Galaxy job execution states. """
    
    NEW = "new"
    RESUBMITTED = "resubmitted"
    UPLOAD = "upload"
    WAITING = "waiting"
    QUEUED = "queued"
    RUNNING = "running"
    OK = "ok"
    ERROR = "error"
    FAILED = "failed"
    PAUSED = "paused"
    DELETING = "deleting"
    DELETED = "deleted"
    STOPPING = "stop"
    STOPPED = "stopped"
    SKIPPED = "skipped"