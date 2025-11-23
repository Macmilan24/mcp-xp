from enum import IntEnum

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


class TTLiveConfig(IntEnum):
    """ Centralizing TTL for invocation and worklows. """
    
    WORKFLOW_CACHE = 600
    INVOCATION_WORKFLOW_MAPPING = 120
    INVOCATION_LIST = 10
    RAW_INVOCATION_LIST = 10
    DUPLICATE_CHECK = 3
    INVOCACTION_RESULT = 259200  # 3 day cache for invocation result.