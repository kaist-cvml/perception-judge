echo "Starting evaluation..."



temperature=1.0
top_p=0.95
top_k=20
temperature=1.0

list="0 1 2"




######### Evaluation for Perception-Judge-Qwen3-4B #########

CKPT_PATH="sjpark5800/Perception-Judge-Qwen3-4B"



for K in $list
do

    echo "🚀 Starting generation... PAIR"

    CUDA_VISIBLE_DEVICES=0,1 python3 generate_judgment.py \
        --ckpt $CKPT_PATH \
        --split "pair" \
        --k $K \
        --top_k $top_k \
        --temperature $temperature \
        --top_p $top_p \
        --filename "Perception-Judge-Qwen3-4B"



    echo "🚀 Starting generation... SCORE"

    CUDA_VISIBLE_DEVICES=0,1 python3 generate_judgment.py \
        --ckpt $CKPT_PATH \
        --split "score" \
        --k $K \
        --top_k $top_k \
        --temperature $temperature \
        --top_p $top_p \
        --filename "Perception-Judge-Qwen3-4B"


    echo "🚀 Starting generation... BATCH"

    CUDA_VISIBLE_DEVICES=0,1 python3 generate_judgment.py \
        --ckpt $CKPT_PATH \
        --split "batch" \
        --k $K \
        --top_k $top_k \
        --temperature $temperature \
        --top_p $top_p \
        --filename "Perception-Judge-Qwen3-4B"


done




######### Evaluation for Qwen3-VL-4B-Thinking #########


CKPT_PATH="Qwen/Qwen3-VL-4B-Thinking"


for K in $list
do

    echo "🚀 Starting generation... PAIR"

    CUDA_VISIBLE_DEVICES=0,1 python3 generate_judgment.py \
        --ckpt $CKPT_PATH \
        --split "pair" \
        --k $K \
        --top_k $top_k \
        --temperature $temperature \
        --top_p $top_p \
        --filename "Qwen3-VL-4B-Thinking"



    echo "🚀 Starting generation... SCORE"

    CUDA_VISIBLE_DEVICES=0,1 python3 generate_judgment.py \
        --ckpt $CKPT_PATH \
        --split "score" \
        --k $K \
        --top_k $top_k \
        --temperature $temperature \
        --top_p $top_p \
        --filename "Qwen3-VL-4B-Thinking"


    echo "🚀 Starting generation... BATCH"

    CUDA_VISIBLE_DEVICES=0,1 python3 generate_judgment.py \
        --ckpt $CKPT_PATH \
        --split "batch" \
        --k $K \
        --top_k $top_k \
        --temperature $temperature \
        --top_p $top_p \
        --filename "Qwen3-VL-4B-Thinking"


done

