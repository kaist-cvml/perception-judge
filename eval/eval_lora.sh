echo "Starting evaluation..."



temperature=1.0
top_p=0.95
top_k=20
temperature=1.0

list="0 1 2"



######### Evaluation for Perception-Judge-Flex-32B w/ LoRA #########

# MUST DO: Run `perception_judge/lora_weight_merge.py` first 
CKPT_PATH="../verl/Flex-VL-32B-Instruct"

LoRA_PATH="sjpark5800/Perception-Judge-Flex-32B-LoRA"



for K in $list
do

    echo "🚀 Starting generation... PAIR"

    CUDA_VISIBLE_DEVICES=4,5,6,7 python3 generate_judgment_w_lora.py \
        --ckpt $CKPT_PATH \
        --split "pair" \
        --k $K \
        --top_k $top_k \
        --temperature $temperature \
        --top_p $top_p \
        --lora_path $LoRA_PATH \
        --filename "Perception-Judge-Flex-32B"



    echo "🚀 Starting generation... SCORE"

    CUDA_VISIBLE_DEVICES=4,5,6,7 python3 generate_judgment_w_lora.py \
        --ckpt $CKPT_PATH \
        --split "score" \
        --k $K \
        --top_k $top_k \
        --temperature $temperature \
        --top_p $top_p \
        --lora_path $LoRA_PATH \
        --filename "Perception-Judge-Flex-32B"


    echo "🚀 Starting generation... BATCH"

    CUDA_VISIBLE_DEVICES=4,5,6,7 python3 generate_judgment_w_lora.py \
        --ckpt $CKPT_PATH \
        --split "batch" \
        --k $K \
        --top_k $top_k \
        --temperature $temperature \
        --top_p $top_p \
        --lora_path $LoRA_PATH \
        --filename "Perception-Judge-Flex-32B"

done

