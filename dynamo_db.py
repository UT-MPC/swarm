# DynamoDB table indices
DEVICE_ID = "DeviceId"
GOAL_DIST = "GoalDist"
LOCAL_DIST = "LocalDist"
DATA_INDICES = "DataIndices"

MODEL_INFO = "ModelInfo"

TIMESTAMPS = "TimeStamps"
EVAL_HIST_LOSS = "EvalHistLoss"
EVAL_HIST_METRIC = "EvalHistMetric"
ENC_IDX = "EncIdx"
HOSTNAME = "Hostname"
TOTAL_ENC_IDX = "TotalEncIdx"

DEV_STATUS = "DeviceStatus"

ERROR_TRACE = "ErrorTrace"

# Overmind worker state DB
WORKER_ID = "WorkerId"
WORKER_STATUS = "WorkerStatus"
WORKER_HISTORY = "WorkerHistory"

# TaskHistory keys
WTIMESTAMP = "TimeStamp"
ACTION_TYPE = "ActionType"
TASK_DETAILS = "TaskDetails"
ERROR_MSG = "ErrorMsg"

# Action Types
WORKER_CREATED = "WorkerCreated"
TASK_START = "TaskStart"
TASK_FAILED = "TaskFailed"
TASK_TIMEOUT = "TaskTimeout"
TASK_END = "TaskEnd"

# Done-tasks table
TASK_ID = "TaskId"
IS_PROCESSED = "IsProcessed"  # is it processed by overmind controller