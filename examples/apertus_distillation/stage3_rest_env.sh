# Settings of the rest-of-Stage-3 cache, for launch_stage3_rest.sh and its preflight:
# top-256 of the sequences that Stage P and the pruning set leave, in shards numbered
# from 0 in a cache of their own, so the finished Stage P cache is never touched.
CACHE_ROOT=${CACHE_ROOT:-/capstor/store/cscs/swissai/infra01/apertus_1p5_distillation_cache}
REST_SHARDS=3013                      # 24,682,496 sequences; 5,589 stay unread
SEQS_PER_SHARD=8192
export BLEND=$CACHE_ROOT/distill/blend.txt    # builds the datasets of Stage P, with its index
export TARGET_TOKENS=$((REST_SHARDS * SEQS_PER_SHARD * 8192))  # sequences of this cache
export HELD_OUT="--distill-heldout-blend $CACHE_ROOT/distill-rest/blend.txt \
  --distill-heldout-after 24051712 \
  --distill-heldout-skip-blend $CACHE_ROOT/pruning/blend.txt --distill-heldout-skip-size 1024"
export MODE=topk SLOTS=256 SEQS_PER_CHUNK=8
export TMPDIR=/tmp  # short: DataLoader workers put their sockets here
export DATASET_CACHE_DIR=$CACHE_ROOT/dataset-index
