CUDA_VISIBLE_DEVICES=0 python src/batchInference.py \
    --model_name_or_path /workspace/mnt/storage/yangdecheng@supremind.com/ydc-wksp-llm3/models/Qwen2-7B \
    --adapter_name_or_path saves/qwen2-7b/lora/sft/qwen2_7b_sft_4_3 \
    --dataset importance_stock_eval20241122_sft \
    --dataset_dir data \
    --template alpaca \
    --finetuning_type lora \
    --lora_target q_proj,v_proj,up_proj \
    --val_size 0.05 \
    --plot_loss \
    --temperature 0.5 \
    --top_k 1 \
    --top_p 0.7



# --model_name_or_path /workspace/mnt/storage/yangdecheng@supremind.com/ydc-wksp-llm3/models/Qwen2-7B \
# --adapter_name_or_path saves/qwen2-7b/lora/sft/qwen2_7b_sft_1 \