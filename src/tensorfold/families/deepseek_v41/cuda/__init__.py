"""DeepSeek-V4.1-Flash on two GPUs: row-invariant kernels, fp32 partials gathered and added in rank order."""

# M:n and K:n in this package cite lines of inference/model.py and inference/kernel.py of
# deepseek-ai/DeepSeek-V4.1-Flash at revision dba1be0

MAX_ROWS = 6                        # a verify window's rows: the target token and up to BLOCK drafts
GRAPH_ROWS = tuple(range(1, MAX_ROWS + 1))
BLOCK = 5                           # rows of one DSpark block (the most drafts a round)
RING = 128                          # sliding-window slots per layer (the checkpoint's window)
PREFILL_ROWS = 2048                 # rows of one prompt chunk
MOE_WINDOW = 1024                   # prompt rows per routed-expert pass
SCORE_BYTES = 1 << 30               # the indexer's score scratch, the cap on its prompt row blocks
DEFAULT_DRAFTS = BLOCK              # the most drafts a round unless --mtp-drafts asks otherwise
DEFAULT_CONFIDENCE = 0.15           # drafts stop under this DSpark confidence product unless a flag sets the policy
MAX_LANES = 4                       # sequences one shared verify forward serves (--parallel at most)
DEFAULT_SHARE = 0.5                 # --parallel 2 or more: rounds take this share of a prompt span's time
