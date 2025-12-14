#! /bin/bash

source .venv/bin/activate
# hf download openai-community/gpt2 --local-dir 
# # hf download Qwen/Qwen2-0.5B --local-dir models/qwen2-0.5b/
# hf download HoangTran223/MCW_KD_Teacher_Mistral7B --local-dir models/teacher_mistral7b/
# hf download mistralai/Mistral-7B-v0.1 --local-dir models/mistral-7b-v0.1/
# hf download TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --local-dir models/tinyllama-1.1b/
# # hf download Qwen/Qwen1.5-1.8B --local-dir models/qwen1.5-1.8b/

# # python -m weighted_ctkd.hf_download mrtuandao/weighted-CTKD\
# #  experiments/sft_qwen1.5-1.8b/20251119_032141/checkpoints/epoch_9\
# #  models/sft_qwen1.5-1.8b/

# python -m weighted_ctkd.hf_download mrtuandao/weighted-CTKD \
#  experiments/tuandao_qwen1.5-1.8b_to_gpt2-120m/20251130_132733/checkpoints/epoch_19 \
#  models/tuandao_qwen1.5-1.8b_to_gpt2-120m/

# # hf download HoangTran223/MCW_KD_GPT2_SFT-1 --local-dir models/mcw_kd_gpt2_sft-1/

# # hf download MiniLLM/SFT-gpt2-120M --local-dir models/minillm_sft_gpt2-120m/


# # BASE LINES
# # gpt2-120m
# hf download HoangTran223/MCW_KD_GPT2_SFT-1 --local-dir MCW_KD_GPT2_SFT-1
# hf download HoangTran223/MCW_KD_GPT2_ULD --local-dir models/uld_gpt2-120m/
# hf download HoangTran223/MCW_KD_GPT2_DSKD --local-dir models/dskd_gpt2-120m/
# hf download HoangTran223/MCW_KD_GPT2_MinED --local-dir models/mined_gpt2-120m/
# hf download HoangTran223/MCW_KD_GPT2_MultiOT --local-dir models/multiot_gpt2-120m/
# hf download HoangTran223/MCW_KD_Teacher_Qwen1.5-1.8B --local-dir models/teacher_qwen1.5-1.8b/

# mini
wget https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors -O model_hub/gpt2/gpt2-base/model.safetensors
wget https://huggingface.co/HoangTran223/MCW_KD_GPT2_SFT-1/resolve/main/pytorch_model.bin -O outputs/gpt2/gpt2-base/hoang_sft/MCW_KD_GPT2_SFT-1/pytorch_model.bin
wget "https://huggingface.co/mrtuandao/weighted-CTKD/resolve/main/gpt2/gpt2-base/sft/criterion%3Dcross_entropy__default-bf16__epoch%3D20__bsz%3D4x2x1%3D8__lr%3D0.0005/epoch20_step28580_loss6.3126_rougel24.6371/pytorch_model.bin" -O outputs/gpt2/gpt2-base/sft/criterion=cross_entropy__default-bf16__epoch=20__bsz=4x2x1=8__lr=0.0005/epoch20_step28580_loss6.3126_rougel24.6371/pytorch_model.bin

hf download HoangTran223/MCW_KD_GPT2_ULD --local-dir outputs/gpt2/gpt2-base/uld/MCW_KD_GPT2_ULD
hf download HoangTran223/MCW_KD_GPT2_DSKD --local-dir outputs/gpt2/gpt2-base/dskd/MCW_KD_GPT2_DSKD
hf download HoangTran223/MCW_KD_GPT2_MinED --local-dir outputs/gpt2/gpt2-base/mined/MCW_KD_GPT2_MinED
hf download HoangTran223/MCW_KD_GPT2_MultiOT --local-dir outputs/gpt2/gpt2-base/multiot/MCW_KD_GPT2_MultiOT

hf upload mrtuandao/weighted-CTKD outputs/ .