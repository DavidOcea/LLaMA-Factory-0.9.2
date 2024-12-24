cd /workspace/mnt/storage/yangdecheng@supremind.com/ydc-wksp-llm3/LLaMA-Factory-main

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install deepspeed==0.14.4
python setup.py install

llamafactory-cli train examples/train_lora/qwen2_7b_lora_ppo.yaml 

