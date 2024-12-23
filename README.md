# Local LLM Comparison & Colab Links (WIP)
(Update Nov. 27, 2023) The original goal of the repo was to compare some smaller models (7B and 13B) that can be run on consumer hardware so every model had a score for a set of questions from GPT-4. But I realized that as there are many more capable models appearing, the evaluation and comparison process may not suffice. 

Therefore, I'm only putting Colab WebUI links for the newer models and you can try them out yourselves with a few clicks - after all, the effectiveness of a language model relies heavily on its suitability for your specific use case. By trying out the models firsthand, you can assess their performance and determine which one best fits your needs.

## Newer Models without Scores:
These models can be run on consumer hardware and are generally good (from Reddit suggestions and my own experience). Try them out yourselves (tap the 'Open in Colab' button)!

| Model                                                 | Link                                                                                                       | Colab Link                                                                                                                                                                                                                                                                                | Date Added | Note                                                                              |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | --------------------------------------------------------------------------------- |
| zephyr-7B-beta-GGUF                                   | [https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF](https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF) | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/zephyr_7B_beta_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                            | 2023/11/27 | Roleplay okay, not censored                                                       |
| OpenHermes-2.5-Mistral-7B-GGUF                        | https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF                                             | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/openhermes_2_5_mistral_7b_Q5_K_M_gguf.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                          | 2023/11/27 | Roleplay good, not censored                                                       |
| dolphin-2.2.1-mistral-7B-GGUF                         | https://huggingface.co/TheBloke/dolphin-2.2.1-mistral-7B-GGUF                                              | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/dolphin_2_2_1_mistral_7B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                  | 2023/11/27 | Roleplay okay, not censored                                                       |
| neural-chat-7B-v3-1-GGUF                              | https://huggingface.co/TheBloke/neural-chat-7B-v3-1-GGUF                                                   | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/neural_chat_7B_v3_1_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                       | 2023/11/27 | Roleplay okay, not censored; some logic flaws                                     |
| openchat_3.5-16k-GGUF                                 | https://huggingface.co/TheBloke/openchat_3.5-16k-GGUF                                                      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/openchat_3_5_16k_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                          | 2023/11/27 | Censored                                                                          |
| Starling-LM-7B-alpha-GGUF                             | https://huggingface.co/TheBloke/Starling-LM-7B-alpha-GGUF                                                  | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Starling_LM_7B_alpha_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                      | 2023/11/29 | Censored; highly rated on Reddit                                                  |
| Orca-2-7B-GGUF                                        | https://huggingface.co/TheBloke/Orca-2-7B-GGUF                                                             | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Orca_2_7B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                                 | 2023/11/29 | Censored                                                                          |
| Orca-2-13B-GGUF                                       | https://huggingface.co/TheBloke/Orca-2-13B-GGUF                                                            | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Orca_2_13B_GGUF.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                               | 2023/11/29 | Censored, some weird logic flaws worse than 7B version                            |
| MythoMist-7B-GGUF                                     | https://huggingface.co/TheBloke/MythoMist-7B-GGUF                                                          | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/MythoMist_7B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                              | 2023/11/29 | Roleplay okay, not censored; some logic flaws                                     |
| NeuralHermes-2.5-Mistral-7B-GGUF                      | https://huggingface.co/TheBloke/NeuralHermes-2.5-Mistral-7B-GGUF                                           | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/NeuralHermes_2_5_Mistral_7B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                               | 2023/12/05 | Roleplay good, not censored                                                       |
| stablelm-zephyr-3b-GGUF                               | https://huggingface.co/TheBloke/stablelm-zephyr-3b-GGUF                                                    | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/stablelm_zephyr_3b_Q5_K_M_gguf.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                 | 2023/12/11 | 3B; roleplay ok; not censored; some logic flaws                                   |
| deepseek-llm-7B-chat-GGUF                             | https://huggingface.co/TheBloke/deepseek-llm-7B-chat-GGUF                                                  | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/deepseek_llm_7B_chat_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                      | 2023/12/11 | Censored                                                                          |
| Mistral-7B-Instruct-v0.2-GGUF                         | https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF                                              | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Mistral_7B_Instruct_v0_2_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                  | 2023/12/13 | Partially censored; role play ok; highly rated on Reddit                          |
| Mixtral-8x7B-Instruct-v0.1-GGUF                       | https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF                                            | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Mixtral_8x7B_Instruct_v0_1_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                | 2023/12/13 | MOE model; partially censored; role play ok                                       |
| deepsex-34b-GGUF                                      | https://huggingface.co/TheBloke/deepsex-34b-GGUF                                                           | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/deepsex_34b_gguf.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                               | 2023/12/14 | 34B; NSFW model                                                                   |
| phi-2                                                 | https://huggingface.co/microsoft/phi-2                                                                     | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/phi_2.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                                          | 2023/12/15 | 2.7B; Base model;                                                                 |
| Xwin-MLewd-13B-v0.2-GGUF                              | https://huggingface.co/TheBloke/Xwin-MLewd-13B-v0.2-GGUF                                                   | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Xwin_MLewd_13B_v0_2_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                       | 2023/12/15 | 13B; NSFW model                                                                   |
| MythoMax-L2-13B-GGUF                                  | https://huggingface.co/TheBloke/MythoMax-L2-13B-GGUF                                                       | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/MythoMax_L2_13B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                           | 2023/12/15 | 13B; Censored; role play ok                                                       |
| LLaMA2-13B-Tiefighter-GGUF                            | https://huggingface.co/TheBloke/LLaMA2-13B-Tiefighter-GGUF                                                 | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/LLaMA2_13B_Tiefighter_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                     | 2023/12/15 | 13B; role play good                                                               |
| LLaMA2-13B-Psyfighter2-GGUF                           | https://huggingface.co/TheBloke/LLaMA2-13B-Psyfighter2-GGUF                                                | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/LLaMA2_13B_Psyfighter2_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                    | 2023/12/15 | 13B; Partially censored; role play ok; recommended on Reddit                      |
| Noromaid-13B-v0.1.1-GGUF                              | https://huggingface.co/TheBloke/Noromaid-13B-v0.1.1-GGUF                                                   | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Noromaid_13B_v0_1_1_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                       | 2023/12/15 | 13B; NSFW model; roleplay good                                                    |
| dolphin-2.5-mixtral-8x7b-GGUF                         | https://huggingface.co/TheBloke/dolphin-2.5-mixtral-8x7b-GGUF                                              | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/dolphin_2_5_mixtral_8x7b_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                  | 2023/12/20 | MOE model; claims to be uncensored but denies some requests; role play not usable |
| SOLAR-10.7B-Instruct-v1.0-GGUF                        | https://huggingface.co/TheBloke/SOLAR-10.7B-Instruct-v1.0-GGUF                                             | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/SOLAR_10_7B_Instruct_v1_0_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                 | 2023/12/21 | 10.7B; Censored; role play ok                                                     |
| Nous-Hermes-2-SOLAR-10.7B-GGUF                        | https://huggingface.co/TheBloke/Nous-Hermes-2-SOLAR-10.7B-GGUF                                             | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Nous_Hermes_2_SOLAR_10_7B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                 | 2024/01/08 | 10.7B; Partially Censored; role play good                                         |
| openchat-3.5-0106-GGUF                                | https://huggingface.co/TheBloke/openchat-3.5-0106-GGUF                                                     | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/openchat_3_5_0106_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                         | 2024/01/12 | Role play and creative writing good; not censored                                 |
| Mistral-7B-Instruct-v0.2-code-ft-GGUF                 | https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-code-ft-GGUF                                      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Mistral_7B_Instruct_v0_2_code_ft_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                          | 2024/01/12 | Coding model                                                                      |
| Panda-7B-v0.1-GGUF                                    | https://huggingface.co/TheBloke/Panda-7B-v0.1-GGUF                                                         | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Panda_7B_v0_1_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                             | 2024/01/12 | Role play ok; creative writing good; partially censored                           |
| LLaMA-Pro-8B-GGUF                                     | https://huggingface.co/TheBloke/LLaMA-Pro-8B-GGUF                                                          | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/LLaMA_Pro_8B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                              | 2024/01/16 | Claims to be good at math/coding; quality so so                                   |
| dolphin-2.6-mistral-7B-dpo-laser-GGUF                 | https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-dpo-laser-GGUF                                      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/dolphin_2_6_mistral_7B_dpo_laser_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                          | 2024/01/16 | Uncensored; Good quality; role play good                                          |
| laserxtral-GGUF                                       | https://huggingface.co/dagbs/laserxtral-GGUF                                                               | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/laserxtral_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                                | 2024/01/17 | 4x7B MOE model; half memory than Mixtral; not censored; role play good            |
| Mixtral_7Bx2_MoE-GGUF                                 | https://huggingface.co/TheBloke/Mixtral_7Bx2_MoE-GGUF                                                      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Mixtral_7Bx2_MoE_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                          | 2024/01/23 | 2x7B MOE model; uncensored; role play ok                                          |
| CapybaraHermes-2.5-Mistral-7B-GGUF                    | https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF                                         | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/CapybaraHermes_2_5_Mistral_7B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                             | 2024/02/06 | Censored;                                                                         |
| stable-code-3b-GGUF                                   | https://huggingface.co/TheBloke/stable-code-3b-GGUF/                                                       | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/stable_code_3b_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                            | 2024/02/06 | 3B; Coding model; quality meh probably too small                                  |
| EstopianMaid-13B-GGUF                                 | https://huggingface.co/TheBloke/EstopianMaid-13B-GGUF                                                      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/EstopianMaid_13B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                          | 2024/02/06 | 13B; Roleplay model                                                               |
| gemma-7b-it-GGUF                                      | https://huggingface.co/mlabonne/gemma-7b-it-GGUF/                                                          | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/gemma_7b_it.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                                    | 2024/02/28 | 7B; Censored                                                                      |
| StarCoder2-15B-GGUF                                   | https://huggingface.co/second-state/StarCoder2-15B-GGUF                                                    | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/StarCoder2_15B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                            | 2024/03/20 | 15B; coding model; highly voted on r/LocalLLaMA                                   |
| Qwen1.5-7B-Chat-GGUF                                  | https://huggingface.co/Qwen/Qwen1.5-7B-Chat-GGUF                                                           | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Qwen1_5_7B_Chat_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                           | 2024/03/20 | 7B; Censored; highly voted on r/LocalLLaMA                                        |
| Qwen1.5-14B-Chat-GGUF                                 | https://huggingface.co/Qwen/Qwen1.5-14B-Chat-GGUF                                                          | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Qwen1_5_14B_Chat_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                          | 2024/03/20 | 14B; Censored; highly voted on r/LocalLLaMA                                       |
| Hermes-2-Pro-Mistral-7B-GGUF                          | https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B-GGUF                                           | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Hermes_2_Pro_Mistral_7B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                   | 2024/03/22 | 7B; Not censored; role play ok; highly ranked on Huggingface                      |
| Nous-Hermes-2-Mistral-7B-DPO-GGUF                     | https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO-GGUF                                      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Nous_Hermes_2_Mistral_7B_DPO_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                              | 2024/03/22 | 7B; Partially censored; role play good; highly ranked on Huggingface              |
| stable-code-instruct-3b-GGUF                          | https://huggingface.co/bartowski/stable-code-instruct-3b-GGUF                                              | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/stable_code_instruct_3b_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                   | 2024/03/27 | 3B; instruction tuned code generation model                                       |
| Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4                      | https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4                                               | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Qwen1_5_MoE_A2_7B_Chat_GPTQ_Int4.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                               | 2024/04/03 | MOE; Small footprint; some logical errors                                         |
| Octopus-v2                                            | https://huggingface.co/NexaAIDev/Octopus-v2/                                                               | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Octopus_v2.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                                     | 2024/04/07 | 2B; non-quantized; optimized for on-device Android APIs                           |
| codegemma-7b-it-GGUF                                  | https://huggingface.co/lmstudio-community/codegemma-7b-it-GGUF                                             | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/codegemma_7b_it_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                           | 2024/04/18 | 7B; coding model                                                                  |
| CodeQwen1.5-7B-Chat-GGUF                              | https://huggingface.co/Qwen/CodeQwen1.5-7B-Chat-GGUF                                                       | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/CodeQwen1_5_7B_Chat_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                       | 2024/04/18 | 7B; coding model                                                                  |
| WizardLM-2-7B-GGUF                                    | https://huggingface.co/MaziyarPanahi/WizardLM-2-7B-GGUF                                                    | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/WizardLM_2_7B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                             | 2024/04/18 | 7B; Censored                                                                      |
| Meta-Llama-3-8B-Instruct-GGUF                         | https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF                                          | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Meta_Llama_3_8B_Instruct_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                  | 2024/04/19 | 8B; Censored                                                                      |
| dolphin-2.9-llama3-8b-gguff                           | https://huggingface.co/cognitivecomputations/dolphin-2.9-llama3-8b-gguff                                   | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/dolphin_2_9_llama3_8b_gguf.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                     | 2024/04/22 | 8B; Uncensored; logic seems degraded from Llama-3-8B                              |
| Lexi-Llama-3-8B-Uncensored-GGUF                       | https://huggingface.co/Orenguteng/Lexi-Llama-3-8B-Uncensored-GGUF                                          | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Lexi_Llama_3_8B_Uncensored_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                | 2024/04/24 | 8B; Uncensored                                                                    |
| Llama3-8B-Chinese-Chat-GGUF                           | https://huggingface.co/QuantFactory/Llama3-8B-Chinese-Chat-GGUF                                            | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Llama3_8B_Chinese_Chat_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                    | 2024/04/24 | 8B; Chinese                                                                       |
| Phi-3-mini-4k-instruct-gguf                           | https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf                                               | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Phi_3_mini_4k_instruct_gguf.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                    | 2024/04/24 | 3.8B; Censored; Fast                                                              |
| Llama-3-8B-Instruct-32k-v0.1-GGUF                     | https://huggingface.co/MaziyarPanahi/Llama-3-8B-Instruct-32k-v0.1-GGUF                                     | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Llama_3_8B_Instruct_32k_v0_1_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                              | 2024/04/25 | 8B; 32K context; good for summarization of long text                              |
| starcoder2-15b-instruct-v0.1-GGUF                     | https://huggingface.co/bartowski/starcoder2-15b-instruct-v0.1-GGUF                                         | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/starcoder2_15b_instruct_v0_1_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                              | 2024/05/06 | 15B; coding model                                                                 |
| Hermes-2-Pro-Llama-3-8B-GGUF                          | https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF                                           | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Hermes_2_Pro_Llama_3_8B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                   | 2024/05/06 | 8B; Partially censored; JSON, Tool Use, etc.                                      |
| Llama-3-ChatQA-1.5-8B-GGUF                            | https://huggingface.co/bartowski/Llama-3-ChatQA-1.5-8B-GGUFF                                               | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Llama_3_ChatQA_1_5_8B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                     | 2024/05/15 | 8B; Uncensored                                                                    |
| Hermes-2-Theta-Llama-3-8B-GGUF                        | https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF                                         | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Hermes_2_Theta_Llama_3_8B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                 | 2024/05/17 | 8B; Censored; JSON, Tool Use, etc                                                 |
| blossom-v5.1-9b-GGUF                                  | https://huggingface.co/bartowski/blossom-v5.1-9b-GGUF                                                      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/blossom_v5_1_9b_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                           | 2024/05/17 | 9B; Wizard/Orca/Math/Chinese/English mixed dataset                                |
| falcon-11B-GGUF                                       | https://huggingface.co/bartowski/falcon-11B-GGUF                                                           | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/falcon_11B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                                | 2024/05/17 | 11B; base raw pretrained model                                                    |
| llama-3-cat-8b-instruct-v1-GGUFF                      | https://huggingface.co/bartowski/llama-3-cat-8b-instruct-v1-GGUFF                                          | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/llama_3_cat_8b_instruct_v1_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                | 2024/05/20 | 8B; partially censored; system instruction fidelity; Role play                    |
| Yi-1.5-9B-Chat-GGUF                                   | https://huggingface.co/bartowski/Yi-1.5-9B-Chat-GGUF                                                       | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Yi_1_5_9B_Chat_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                            | 2024/05/20 | 9B; Censored                                                                      |
| SFR-Iterative-DPO-LLaMA-3-8B-R-GGUF                   | https://huggingface.co/bartowski/SFR-Iterative-DPO-LLaMA-3-8B-R-GGUF                                       | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/SFR_Iterative_DPO_LLaMA_3_8B_R_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                            | 2024/05/22 | 8B; Partially censored                                                            |
| Llama-3-Lumimaid-8B-v0.1-OAS-GGUF-IQ-Imatrix          | https://huggingface.co/Lewdiculous/Llama-3-Lumimaid-8B-v0.1-OAS-GGUF-IQ-Imatrix                            | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Llama_3_Lumimaid_8B_v0_1_OAS_GGUF_IQ_Imatrix.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                   | 2024/05/22 | 8B; Roleplay; Trained to not decline requests; Seems to be trained on Quora data  |
| Mistral-7B-Instruct-v0.3-GGUF                         | https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF                                         | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Mistral_7B_Instruct_v0_3_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                  | 2024/05/23 | 7B; Censored                                                                      |
| L3-8B-Stheno-v3.1-GGUF-IQ-Imatrix                     | https://huggingface.co/Lewdiculous/L3-8B-Stheno-v3.1-GGUF-IQ-Imatrix                                       | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/L3_8B_Stheno_v3_1_GGUF_IQ_Imatrix.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                              | 2024/05/30 | 8B; trained for 1 on 1 uncensored roleplay                                        |
| aya-23-8B-GGUF                                        | https://huggingface.co/bartowski/aya-23-8B-GGUF                                                            | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/aya_23_8B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                                 | 2024/05/30 | 8B; Censored                                                                      |
| LLaMA3-iterative-DPO-final-GGUF                       | https://huggingface.co/bartowski/LLaMA3-iterative-DPO-final-GGUF                                           | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/LLaMA3_iterative_DPO_final_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                | 2024/05/30 | 8B; Censored                                                                      |
| openchat-3.6-8b-20240522-GGUF                         | https://huggingface.co/bartowski/openchat-3.6-8b-20240522-GGUF                                             | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/openchat_3_6_8b_20240522_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                  | 2024/06/04 | 8B; Partially censored                                                            |
| Meta-Llama-3-8B-Instruct-abliterated-v3-GGUF          | https://huggingface.co/failspy/Meta-Llama-3-8B-Instruct-abliterated-v3-GGUF                                | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Meta_Llama_3_8B_Instruct_abliterated_v3_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                   | 2024/06/04 | 8B; Uncensored;                                                                   |
| NeuralDaredevil-8B-abliterated-GGUF                   | https://huggingface.co/QuantFactory/NeuralDaredevil-8B-abliterated-GGUF                                    | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/NeuralDaredevil_8B_abliterated_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                            | 2024/06/19 | 8B; Uncensored;                                                                   |
| Qwen2-7B-Instruct-GGUF                                | https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF                                                         | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Qwen2_7B_Instruct_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                         | 2024/06/24 | 7B; Censored                                                                      |
| DeepSeek-Coder-V2-Lite-Instruct-GGUF                  | https://huggingface.co/lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF                             | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/DeepSeek_Coder_V2_Lite_Instruct_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                           | 2024/06/27 | 16B; Instruct coding model                                                        |
| internlm2_5-7b-chat-gguf                              | https://huggingface.co/internlm/internlm2_5-7b-chat-gguf                                                   | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/internlm2_5_7b_chat_gguf.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                       | 2024/07/11 | 7B; Censored; Long context; Reasoning and Tool Use;                               |
| gemma-2-9b-it-GGUF                                    | https://huggingface.co/bartowski/gemma-2-9b-it-GGUF                                                        | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/gemma_2_9b_it_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                             | 2024/07/11 | 9B; Censored                                                                      |
| Smegmma-Deluxe-9B-v1-GGUF                             | https://huggingface.co/bartowski/Smegmma-Deluxe-9B-v1-GGUF                                                 | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Smegmma_Deluxe_9B_v1_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                      | 2024/07/22 | 9B; Less censored gemma (still refuses some requests); Role play                  |
| h2o-danube3-4b-chat-GGUF                              | https://huggingface.co/h2oai/h2o-danube3-4b-chat-GGUF                                                      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/h2o_danube3_4b_chat_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                       | 2024/07/17 | 4B; Partially censored                                                            |
| Tiger-Gemma-9B-v1-GGUF                                | https://huggingface.co/bartowski/Tiger-Gemma-9B-v1-GGUF                                                    | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Tiger_Gemma_9B_v1_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                         | 2024/07/17 | 9B; Uncensored; but appears to have some flaws in logic                           |
| Gemmasutra-9B-v1-GGUF                                 | https://huggingface.co/TheDrummer/Gemmasutra-9B-v1-GGUF                                                    | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Gemmasutra_9B_v1_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                          | 2024/07/24 | 9B; Roleplay with some censorship                                                 |
| Meta-Llama-3.1-8B-Instruct-GGUF                       | https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF                                  | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Meta_Llama_3_1_8B_Instruct_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                | 2024/07/25 | 8B; Censored                                                                      |
| Mistral-Nemo-Instruct-2407-GGUF                       | https://huggingface.co/second-state/Mistral-Nemo-Instruct-2407-GGUF                                        | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Mistral_Nemo_Instruct_2407_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                | 2024/07/25 | 12.2B; Partially Uncensored; Recommended by Redditors                             |
| Celeste-12B-V1.6_iMatrix_GGUF                         | https://huggingface.co/MarsupialAI/Celeste-12B-V1.6_iMatrix_GGUF                                           | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Celeste_12B_V1_6_iMatrix_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                  | 2024/07/30 | 12B; Roleplay & story writing model, Uncensored                                   |
| Hermes-3-Llama-3.1-8B-GGUF                            | https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B-GGUF                                             | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Hermes_3_Llama_3_1_8B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                     | 2024/08/19 | 8B; Uncensored                                                                    |
| Gemma-2-9B-It-SPPO-Iter3-GGUF                         | https://huggingface.co/bartowski/Gemma-2-9B-It-SPPO-Iter3-GGUF                                             | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Gemma_2_9B_It_SPPO_Iter3_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                  | 2024/08/19 | 9B; Recommended on r/LocalLLaMA for summarization                                 |
| Llama-3.1-Storm-8B-GGUF                               | https://huggingface.co/bartowski/Llama-3.1-Storm-8B-GGUF                                                   | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Llama_3_1_Storm_8B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                        | 2024/08/20 | 8B; Censored                                                                      |
| Phi-3.5-mini-instruct-GGUF                            | https://huggingface.co/lmstudio-community/Phi-3.5-mini-instruct-GGUF                                       | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/d41883e2bb60bd566483990ceac554f9e56c4207/Phi_3_5_mini_instruct_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | 2024/08/21 | 3.5B; Censored, small & fast                                                      |
| Phi-3.5-mini-instruct_Uncensored-GGUF                 | https://huggingface.co/bartowski/Phi-3.5-mini-instruct_Uncensored-GGUF                                     | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Phi_3_5_mini_instruct_Uncensored_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                          | 2024/08/26 | 3.5B; Uncensored, small & fast, but has logic flaws & inconsistent results        |
| NemoMix-Unleashed-12B-GGUF                            | https://huggingface.co/bartowski/NemoMix-Unleashed-12B-GGUF                                                | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/NemoMix_Unleashed_12B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                     | 2024/08/26 | 12B; Partially censored, Roleplaying & story telling                              |
| Mistral-NeMo-Minitron-8B-Base-GGUF                    | https://huggingface.co/QuantFactory/Mistral-NeMo-Minitron-8B-Base-GGUF                                     | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Mistral_NeMo_Minitron_8B_Base_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                             | 2024/08/26 | 8B; Base model                                                                    |
| Yi-Coder-9B-Chat-GGUF                                 | https://huggingface.co/bartowski/Yi-Coder-9B-Chat-GGUF                                                     | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Yi_Coder_9B_Chat_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                          | 2024/09/05 | 9B; Instruction coding model                                                      |
| Llama-3.2-3B-Instruct-Q8_0-GGUF                       | https://huggingface.co/hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF                                      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Llama_3_2_3B_Instruct_Q8_0_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                | 2024/10/01 | 3B; Small censored model                                                          |
| Qwen2.5-7B-Instruct-GGUF                              | https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF                                                  | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Qwen2_5_7B_Instruct_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                       | 2024/10/01 | 7B; Censored, recommended on r/LocalLLaMA                                         |
| Qwen2.5-Coder-7B-Instruct-GGUF                        | https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF                                            | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Qwen2_5_Coder_7B_Instruct_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                 | 2024/10/01 | 7B; Instruction coding model, recommended on r/LocalLLaMA                         |
| Llama-3.1-8B-Lexi-Uncensored-V2-GGUF                  | https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF                                     | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Llama_3_1_8B_Lexi_Uncensored_V2_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                           | 2024/10/15 | 8B; Partially censored                                                            |
| SuperNova-Medius-GGUF                                 | https://huggingface.co/arcee-ai/SuperNova-Medius-GGUF                                                      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/SuperNova_Medius_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                          | 2024/10/15 | 14B; Censored, recommended on on r/LocalLLaMA                                     |
| Llama-3.2-3B-Instruct-uncensored-GGUF                 | https://huggingface.co/mradermacher/Llama-3.2-3B-Instruct-uncensored-GGUF                                  | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Llama_3_2_3B_Instruct_uncensored_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                          | 2024/10/15 | 3B; Small, uncensored                                                             |
| Ministral-8B-Instruct-2410-HF-GGUF-TEST               | https://huggingface.co/bartowski/Ministral-8B-Instruct-2410-HF-GGUF-TEST                                   | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Ministral_8B_Instruct_2410_HF_GGUF_TEST.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                        | 2024/10/21 | 8B; Partially censored                                                            |
| granite-3.0-8b-instruct-GGUF                          | https://huggingface.co/bartowski/granite-3.0-8b-instruct-GGUF                                              | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/granite_3_0_8b_instruct_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                   | 2024/10/28 | 8B; Censored                                                                      |
| aya-expanse-8b-GGUF                                   | https://huggingface.co/bartowski/aya-expanse-8b-GGUF                                                       | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/aya_expanse_8b_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                            | 2024/10/28 | 8B; Censored                                                                      |
| Qwen2.5-Coder-14B-Instruct-GGUF                       | https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF                                                | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Qwen2_5_Coder_14B_Instruct_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                | 2024/11/12 | 14B; Coding model, good at its size                                               |
| Marco-o1-GGUF                                         | https://huggingface.co/bartowski/Marco-o1-GGUF                                                             | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Marco_o1_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                                  | 2024/11/25 | 7B; CoT model to mimic o1 for reasoning                                           |
| Llama-Sentient-3.2-3B-Instruct-GGUF                   | https://huggingface.co/prithivMLmods/Llama-Sentient-3.2-3B-Instruct-GGUF                                   | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Llama_Sentient_3_2_3B_Instruct_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                            | 2024/11/25 | 3B; Censored                                                                      |
| Gemma-The-Writer-N-Restless-Quill-10B-Uncensored-GGUF | https://huggingface.co/DavidAU/Gemma-The-Writer-N-Restless-Quill-10B-Uncensored-GGUF                       | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Gemma_The_Writer_N_Restless_Quill_10B_Uncensored_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>          | 2024/12/05 | 10B; Uncensored story writing model                                               |
| INTELLECT-1-Instruct-GGUF                             | https://huggingface.co/lmstudio-community/INTELLECT-1-Instruct-GGUF                                        | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/INTELLECT_1_Instruct_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                      | 2024/12/05 | 10B; Censored                                                                     |
| L3.1-RP-Hero-BigTalker-8B-GGUF                        | https://huggingface.co/DavidAU/L3.1-RP-Hero-BigTalker-8B-GGUF                                              | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/L3_1_RP_Hero_BigTalker_8B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                 | 2024/12/05 | 8B; Uncensored NSFW roleplay model                                                |
| EXAONE-3.5-7.8B-Instruct-GGUF                         | https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF                                           | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/EXAONE_3_5_7_8B_Instruct_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                  | 2024/12/09 | 7.8B; Censored, English and Korean, context up to 32K                             |
| Virtuoso-Small-GGUF                                   | https://huggingface.co/bartowski/Virtuoso-Small-GGUF                                                       | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Virtuoso_Small_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                            | 2024/12/10 | 14B; Censored                                                                     |
| phi-4                                                 | https://huggingface.co/matteogeniaccio/phi-4                                                               | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/phi_4.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                                          | 2024/12/16 | 14B; Censored                                                                     |
| Hermes-3-Llama-3.2-3B-GGUF                            | https://huggingface.co/NousResearch/Hermes-3-Llama-3.2-3B-GGUF                                             | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Hermes_3_Llama_3_2_3B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                     | 2024/12/21 | 3B; Partially censored                                                            |
| Falcon3-Mamba-7B-Instruct-GGUF                        | https://huggingface.co/tiiuae/Falcon3-Mamba-7B-Instruct-GGUF                                               | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Falcon3_Mamba_7B_Instruct_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                 | 2024/12/21 | 7B; Mamba1 architecture, 32K context                                              |
| QwQ-LCoT-7B-Instruct-GGUF                             | https://huggingface.co/prithivMLmods/QwQ-LCoT-7B-Instruct-GGUF                                             | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/QwQ_LCoT_7B_Instruct_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                      | 2024/12/21 | 7B; finetuned on CoT                                                              |

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Troyanovsky/Local-LLM-Comparison-Colab-UI&type=Date)](https://star-history.com/#Troyanovsky/Local-LLM-Comparison-Colab-UI&Date)

## Share This Project

If you find this project helpful, please share it:

### Quick Share Links:
- **[Facebook](https://www.facebook.com/sharer.php?u=https%3A%2F%2Fgithub.com%2FTroyanovsky%2FLocal-LLM-Comparison-Colab-UI)**  
- **[Reddit](https://reddit.com/submit?url=https%3A%2F%2Fgithub.com%2FTroyanovsky%2FLocal-LLM-Comparison-Colab-UI&title=Run%20Local%20LLMs%20Easily%20with%20Google%20Colab%21)**  
- **[Twitter](https://twitter.com/intent/tweet?url=https%3A%2F%2Fgithub.com%2FTroyanovsky%2FLocal-LLM-Comparison-Colab-UI&text=Run%20Local%20LLMs%20Easily%20with%20Google%20Colab%21&via=Troyanovsky&hashtags=LLMs%2CAI%2COpenSource)**  
- **[LinkedIn](https://www.linkedin.com/sharing/share-offsite/?url=https%3A%2F%2Fgithub.com%2FTroyanovsky%2FLocal-LLM-Comparison-Colab-UI)**  
- **[VK](http://vk.com/share.php?url=https%3A%2F%2Fgithub.com%2FTroyanovsky%2FLocal-LLM-Comparison-Colab-UI&title=Run%20Local%20LLMs%20Easily%20with%20Google%20Colab%21&comment=Explore%20accessible%20LLMs%20for%20your%20needs%21)**  
- **[Email](mailto:?subject=Run%20Local%20LLMs%20Easily%20with%20Google%20Colab%21&body=Check%20out%20this%20project%3A%20https%3A%2F%2Fgithub.com%2FTroyanovsky%2FLocal-LLM-Comparison-Colab-UI%20It%20makes%20running%20LLMs%20accessible%20and%20easy%20using%20Google%20Colab.)**

<details>

<summary> Older Models with Scores </summary>

## Older Models with Scores:
These models work better among the models I tested on my hardware (i5-12490F, 32GB RAM, RTX 3060 Ti GDDR6X 8GB VRAM):
(Note: Because llama.cpp has made some breaking changes to the support of older ggml models. Some older ggml versions listed below may not work properly on current llama.cpp. But there should be GPTQ equivalents or newer ggml versions for the models.)

NOTE:
- Major edit on June 30, 2023. Since I noticed GPT-4 started to give more granular scores and tends to give higher scores, I have decided to re-run the tests for all models so they can be compared (All evaluations done with GPT-4-0613 for consistency). The scores in the table below are the average of the scores from the latest tests. The scores in the table below are not comparable to the scores in the previous versions of this README.

| Model_Name                                                                      | Avg_Score | Colab_Link                                                                                                                                                                                                                                                    | Date_Added | Link                                                                    |
| :------------------------------------------------------------------------------ | :-------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :--------- | :---------------------------------------------------------------------- |
| Mistral-7B-OpenOrca (using oobabooga/text-generation-webui)                     | 10.00     | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Mistral_7B_OpenOrca.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                | 2023/10/08 | https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF                |
| Llama-2-13B-chat (using oobabooga/text-generation-webui)                        | 9.65      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/Llama_2_13B_chat.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                            | 2023/07/20 | https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML                   |
| wizard-vicuna-13B.ggml.q4_0 (using llama.cpp)                                   | 9.63      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/wizard_vicuna_13B_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                      | 2023/05/07 | https://huggingface.co/TheBloke/wizard-vicuna-13B-GGML                  |
| Nous-Capybara-7B (using oobabooga/text-generation-webui)                        | 9.56      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Nous_Capybara_7B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>              | 2023/10/08 | https://huggingface.co/TheBloke/Nous-Capybara-7B-GGUF                   |
| vicuna-13B-v1.5 (using oobabooga/text-generation-webui)                         | 9.53      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/vicuna_13B_v1_5.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                    | 2023/08/09 | https://huggingface.co/TheBloke/vicuna-13B-v1.5-GGML                    |
| wizardLM-13B-1.0-GPTQ (using oobabooga/text-generation-webui)                   | 9.53      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/wizardLM_13B_1_0_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                       | 2023/05/29 | https://huggingface.co/TheBloke/wizardLM-13B-1.0-GPTQ                   |
| airoboros-13B-gpt4-1.4-GPTQ (using oobabooga/text-generation-webui)             | 9.50      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/airoboros_13B_gpt4_1_4_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                 | 2023/06/30 | https://huggingface.co/TheBloke/airoboros-13B-gpt4-1.4-GPTQ             |
| Nous-Hermes-13B-GPTQ (using oobabooga/text-generation-webui)                    | 9.44      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/Nous_Hermes_13B_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                        | 2023/06/03 | https://huggingface.co/TheBloke/Nous-Hermes-13B-GPTQ/tree/main          |
| Dolphin-Llama-13B (using oobabooga/text-generation-webui)                       | 9.38      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/Dolphin_Llama_13B.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                           | 2023/07/24 | https://huggingface.co/TheBloke/Dolphin-Llama-13B-GGML                  |
| Mistral-7B-Instruct-v0.1 (using oobabooga/text-generation-webui)                | 9.37      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Mistral_7B_Instruct_v0_1_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>      | 2023/10/08 | https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF                  |
| OpenOrca-Platypus2-13B (using oobabooga/text-generation-webui)                  | 9.37      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/OpenOrca_Platypus2_13B.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>             | 2023/08/15 | https://huggingface.co/TheBloke/OpenOrca-Platypus2-13B-GGML             |
| airoboros-l2-13b-gpt4-2.0 (using oobabooga/text-generation-webui)               | 9.34      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/airoboros_l2_13b_gpt4_2_0.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>          | 2023/08/01 | https://huggingface.co/TheBloke/airoboros-l2-13b-gpt4-2.0-GGML          |
| Chronos-13B-v2 (using oobabooga/text-generation-webui)                          | 9.31      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Chronos_13B_v2.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                     | 2023/08/09 | https://huggingface.co/TheBloke/Chronos-13B-v2-GGML                     |
| vicuna-13b-v1.3.0-GPTQ (using oobabooga/text-generation-webui)                  | 9.31      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/vicuna_13b_v1_3_0_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                      | 2023/06/29 | https://huggingface.co/TheBloke/vicuna-13b-v1.3.0-GPTQ                  |
| MythoLogic-13B (using oobabooga/text-generation-webui)                          | 9.31      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/MythoLogic_13B.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                              | 2023/07/20 | https://huggingface.co/TheBloke/MythoLogic-13B-GGML                     |
| Selfee-13B-GPTQ (using oobabooga/text-generation-webui)                         | 9.28      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/Selfee_13B_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                             | 2023/06/07 | https://huggingface.co/TheBloke/Selfee-13B-GPTQ                         |
| WizardLM-13B-V1.2 (using oobabooga/text-generation-webui)                       | 9.28      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/WizardLM_13B_V1_2.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                  | 2023/07/26 | https://huggingface.co/TheBloke/WizardLM-13B-V1.2-GGML                  |
| minotaur-13B-GPTQ (using oobabooga/text-generation-webui)                       | 9.28      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/minotaur_13B_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                           | 2023/06/09 | https://huggingface.co/TheBloke/minotaur-13B-GPTQ                       |
| Pygmalion-2-13B-SuperCOT2 (using oobabooga/text-generation-webui)               | 9.20      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Pygmalion_2_13B_SuperCOT2_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>     | 2023/09/21 | https://huggingface.co/TheBloke/Pygmalion-2-13B-SuperCOT2-GGUF          |
| Athena-v1 (using oobabooga/text-generation-webui)                               | 9.19      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Athena_v1_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                     | 2023/08/31 | https://huggingface.co/TheBloke/Athena-v1-GGUF                          |
| PuddleJumper-13B (using oobabooga/text-generation-webui)                        | 9.1875    | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/PuddleJumper_13B.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                   | 2023/08/29 | https://huggingface.co/TheBloke/PuddleJumper-13B-GGUF                   |
| Nous-Hermes-Llama2 (using oobabooga/text-generation-webui)                      | 9.17      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/Nous_Hermes_Llama2.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                          | 2023/07/24 | https://huggingface.co/TheBloke/Nous-Hermes-Llama2-GGML                 |
| Luban-13B (using oobabooga/text-generation-webui)                               | 9.15      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Luban_13B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                     | 2023/08/31 | https://huggingface.co/TheBloke/Luban-13B-GGUF                          |
| MythoBoros-13B (using oobabooga/text-generation-webui)                          | 9.15      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/MythoBoros_13B.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                              | 2023/07/24 | https://huggingface.co/TheBloke/MythoBoros-13B-GGML                     |
| 13B-Ouroboros (using oobabooga/text-generation-webui)                           | 9.11      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/13B_Ouroboros.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                      | 2023/07/27 | https://huggingface.co/TheBloke/13B-Ouroboros-GGML                      |
| tulu-13B-GPTQ (using oobabooga/text-generation-webui)                           | 9.06      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/tulu_13B_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                               | 2023/06/13 | https://huggingface.co/TheBloke/tulu-13B-GPTQ                           |
| AlpacaCielo-13B (using oobabooga/text-generation-webui)                         | 9.03125   | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/AlpacaCielo_13B.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                    | 2023/07/27 | https://huggingface.co/TheBloke/AlpacaCielo-13B-GGML                    |
| StableBeluga-13B (using oobabooga/text-generation-webui)                        | 9         | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/StableBeluga_13B.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                   | 2023/08/01 | https://huggingface.co/TheBloke/StableBeluga-13B-GGML                   |
| Chronos-Hermes-13B-v2 (using oobabooga/text-generation-webui)                   | 8.97      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Chronos_Hermes_13B_v2.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>              | 2023/08/10 | https://huggingface.co/TheBloke/Chronos-Hermes-13B-v2-GGML              |
| OpenBuddy-Llama2-13B-v11.1 (using oobabooga/text-generation-webui)              | 8.97      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/OpenBuddy_Llama2_13B_v11_1.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>         | 2023/09/05 | https://huggingface.co/TheBloke/OpenBuddy-Llama2-13B-v11.1-GGUF         |
| Camel-Platypus2-13B (using oobabooga/text-generation-webui)                     | 8.94      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Camel_Platypus2_13B.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                | 2023/08/15 | https://huggingface.co/TheBloke/Camel-Platypus2-13B-GGML                |
| airoboros-l2-13b-gpt4-m2.0 (using oobabooga/text-generation-webui)              | 8.94      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/airoboros_l2_13b_gpt4_m2_0_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>    | 2023/09/21 | https://huggingface.co/TheBloke/airoboros-l2-13b-gpt4-m2.0-GGUF         |
| UltraLM-13B-GPTQ (using oobabooga/text-generation-webui)                        | 8.89      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/UltraLM_13B_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                            | 2023/06/30 | https://huggingface.co/TheBloke/UltraLM-13B-GPTQ                        |
| 13B-HyperMantis_GPTQ (using oobabooga/text-generation-webui)                    | 8.88      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/13B_HyperMantis_GPTQ_4bit_128g.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>              | 2023/06/03 | https://huggingface.co/digitous/13B-HyperMantis_GPTQ_4bit-128g/         |
| Stable-Platypus2-13B (using oobabooga/text-generation-webui)                    | 8.875     | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Stable_Platypus2_13B.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>               | 2023/08/15 | https://huggingface.co/TheBloke/Stable-Platypus2-13B-GGML               |
| Airoboros-13B-GPTQ-4bit (using oobabooga/text-generation-webui)                 | 8.84      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/airoboros_13B_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                          | 2023/05/25 | https://huggingface.co/TheBloke/airoboros-13B-GPTQ                      |
| Kuchiki-1.1-L2-7B (using oobabooga/text-generation-webui)                       | 8.84      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Kuchiki_1_1_L2_7B_GGUF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>             | 2023/09/21 | https://huggingface.co/TheBloke/Kuchiki-1.1-L2-7B-GGUF                  |
| WizardLM-1.0-Uncensored-Llama2-13B (using oobabooga/text-generation-webui)      | 8.80625   | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/WizardLM_1_0_Uncensored_Llama2_13B.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | 2023/08/09 | https://huggingface.co/TheBloke/WizardLM-1.0-Uncensored-Llama2-13B-GGML |
| Chronos-Beluga-v2-13B (using oobabooga/text-generation-webui)                   | 8.75      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Chronos_Beluga_v2_13B.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>              | 2023/08/10 | https://huggingface.co/TheBloke/Chronos-Beluga-v2-13B-GGML              |
| Vicuna-13B-CoT-GPTQ (using oobabooga/text-generation-webui)                     | 8.75      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/Vicuna_13B_CoT_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                         | 2023/06/09 | https://huggingface.co/TheBloke/Vicuna-13B-CoT-GPTQ                     |
| wizardLM-7B.q4_2 (in GPT4All)                                                   | 8.75      | No                                                                                                                                                                                                                                                            | 2023/05/07 | https://gpt4all.io/models/ggml-wizardLM-7B.q4_2.bin                     |
| OpenChat_v3.2 (using oobabooga/text-generation-webui)                           | 8.71875   | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/OpenChat_v3_2.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                      | 2023/08/01 | https://huggingface.co/TheBloke/OpenChat_v3.2-GGML                      |
| Huginn-13B (using oobabooga/text-generation-webui)                              | 8.7125    | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Huginn_13B.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                         | 2023/08/10 | https://huggingface.co/TheBloke/Huginn-13B-GGML                         |
| WizardLM-13B-V1.1 (using oobabooga/text-generation-webui)                       | 8.66      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/WizardLM_13B_V1_1_GGML.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                      | 2023/07/17 | https://huggingface.co/TheBloke/WizardLM-13B-V1.1-GGML                  |
| robin-13B-v2-GPTQ (using oobabooga/text-generation-webui)                       | 8.66      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/robin_13B_v2_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                           | 2023/06/19 | https://huggingface.co/TheBloke/robin-13B-v2-GPTQ                       |
| llama-2-13B-Guanaco-QLoRA (using oobabooga/text-generation-webui)               | 8.625     | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/llama_2_13B_Guanaco_QLoRA.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                   | 2023/07/21 | https://huggingface.co/TheBloke/llama-2-13B-Guanaco-QLoRA-GGML          |
| mpt-7b-chat (in GPT4All)                                                        | 8.53      | No                                                                                                                                                                                                                                                            | 2023/05/11 | https://gpt4all.io/models/ggml-mpt-7b-chat.bin                          |
| chronos-hermes-13B-GPTQ (using oobabooga/text-generation-webui)                 | 8.48125   | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/chronos_hermes_13B_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                     | 2023/06/16 | https://huggingface.co/TheBloke/chronos-hermes-13B-GPTQ                 |
| Luna-AI-Llama2-Uncensored (using oobabooga/text-generation-webui)               | 8.46875   | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/Luna_AI_Llama2_Uncensored.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                   | 2023/07/20 | https://huggingface.co/TheBloke/Luna-AI-Llama2-Uncensored-GGML          |
| stable-vicuna-13B-GPTQ-4bit-128g (using oobabooga/text-generation-webui)        | 8.25      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/stable_vicuna_13B_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                      | 2023/05/12 | https://huggingface.co/TheBloke/stable-vicuna-13B-GPTQ                  |
| manticore_13b_chat_pyg_GPTQ (using oobabooga/text-generation-webui)             | 8.21875   | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/manticore_13b_chat_pyg_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                 | 2023/05/24 | https://huggingface.co/TheBloke/manticore-13b-chat-pyg-GPTQ             |
| CAMEL_13B_Combined_Data_GPTQ (using oobabooga/text-generation-webui)            | 8.09375   | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/CAMEL_13B_Combined_Data_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                | 2023/06/10 | https://huggingface.co/TheBloke/CAMEL-13B-Combined-Data-GPTQ            |
| WizardLM-Uncensored-Falcon-7B-GPTQ   (using oobabooga/text-generation-webui)    | 8.09375   | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/WizardLM_Uncensored_Falcon_7B_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>          | 2023/06/02 | https://huggingface.co/TheBloke/WizardLM-Uncensored-Falcon-7B-GPTQ      |
| llama-13b-supercot-GGML (using oobabooga/text-generation-webui)                 | 8.01      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/llama_13b_supercot_GGML.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                     | 2023/07/05 | https://huggingface.co/TheBloke/llama-13b-supercot-GGML                 |
| Project-Baize-v2-13B-GPTQ (using oobabooga/text-generation-webui)               | 7.96875   | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/Project_Baize_v2_13B_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                   | 2023/05/24 | https://huggingface.co/TheBloke/Project-Baize-v2-13B-GPTQ               |
| koala-13B-4bit-128g.GGML (using llama.cpp)                                      | 7.9375    | No                                                                                                                                                                                                                                                            | 2023/05/07 | https://huggingface.co/TheBloke/koala-13B-GPTQ-4bit-128g-GGML           |
| wizard-lm-uncensored-13b-GPTQ-4bit-128g (using oobabooga/text-generation-webui) | 7.90625   | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/wizard_lm_uncensored_13b_GPTQ_4bit_128g.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>     | 2023/05/19 | https://huggingface.co/4bit/WizardLM-13B-Uncensored-4bit-128g           |
| vicuna-7B-v1.3-GPTQ (using oobabooga/text-generation-webui)                     | 7.875     | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/vicuna_7B_v1_3_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                         | 2023/06/29 | https://huggingface.co/TheBloke/vicuna-7B-v1.3-GPTQ                     |
| Manticore-13B-GPTQ (using oobabooga/text-generation-webui)                      | 7.78125   | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/Manticore_13B_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                          | 2023/05/23 | https://huggingface.co/TheBloke/Manticore-13B-GPTQ                      |
| vicuna-13b-1.1-q4_2 (in GPT4All)                                                | 7.75      | No                                                                                                                                                                                                                                                            | 2023/05/07 | https://gpt4all.io/models/ggml-vicuna-13b-1.1-q4_2.bin                  |
| falcon-7b-instruct-GPTQ  (using oobabooga/text-generation-webui)                | 7.625     | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/falcon_7b_instruct_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                     | 2023/06/02 | https://huggingface.co/TheBloke/falcon-7b-instruct-GPTQ                 |
| guanaco-13B-GPTQ (using oobabooga/text-generation-webui)                        | 7.5625    | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/guanaco_13B_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                            | 2023/05/26 | https://huggingface.co/TheBloke/guanaco-13B-GPTQ                        |
| Mythical-Destroyer-V2-L2-13B (using oobabooga/text-generation-webui)            | 7.31      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Mythical_Destroyer_V2_L2_13B.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>       | 2023/08/31 | https://huggingface.co/TheBloke/Mythical-Destroyer-V2-L2-13B-GGUF       |
| Kimiko-v2-13B (using oobabooga/text-generation-webui)                           | 7.25      | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Kimiko_v2_13B.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                     | 2023/08/31 | https://huggingface.co/TheBloke/Kimiko-v2-13B-GGUF                      |
| orca-mini-13b.ggmlv3.q5_K_M (using oobabooga/text-generation-webui)             | 7.0875    | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/orca_mini_13b_ggmlv3_q5_K_M.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                 | 2023/06/28 | https://huggingface.co/TheBloke/orca_mini_13B-GGML                      |
| Platypus2-13B (using oobabooga/text-generation-webui)                           | 7.03125   | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Platypus2_13B.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                      | 2023/08/15 | https://huggingface.co/TheBloke/Platypus2-13B-GGML                      |
| Redmond-Puffin-13B (using oobabooga/text-generation-webui)                      | 7.03125   | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/Redmond_Puffin_13B.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                          | 2023/07/20 | https://huggingface.co/TheBloke/Redmond-Puffin-13B-GGML                 |
| 13B-BlueMethod (using oobabooga/text-generation-webui)                          | 7.025     | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/13B_BlueMethod.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                              | 2023/07/24 | https://huggingface.co/TheBloke/13B-BlueMethod-GGML                     |
| mpt-7b-instruct                                                                 | 6.6875    | No                                                                                                                                                                                                                                                            | 2023/05/12 | https://huggingface.co/TheBloke/MPT-7B-Instruct-GGML                    |
| Kimiko-13B (using oobabooga/text-generation-webui)                              | 6.46875   | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/Kimiko_13B.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                         | 2023/08/01 | https://huggingface.co/TheBloke/Kimiko-13B-GGML                         |
| gpt4-x-alpaca-13b-ggml-q4_0 (using llama.cpp)                                   | 6.0625    | No                                                                                                                                                                                                                                                            | 2023/05/07 | https://huggingface.co/Bradarr/gpt4-x-alpaca-13b-native-ggml-model-q4_0 |
| minotaur-15B-GPTQ (using oobabooga/text-generation-webui)                       | 5.9375    | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/minotaur_15B_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                           | 2023/06/26 | https://huggingface.co/TheBloke/minotaur-15B-GPTQ                       |
| baichuan-vicuna-7B-GGML (using oobabooga/text-generation-webui)                 | 5.90625   | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/baichuan_vicuna_7B_GGML.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                     | 2023/07/05 | https://huggingface.co/TheBloke/baichuan-vicuna-7B-GGML                 |
| gpt4all-j-v1.3-groovy (in GPT4All)                                              | 5.6875    | No                                                                                                                                                                                                                                                            | 2023/05/07 | https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin                |


Many thanks to:  
❤️ GPT4ALl: https://github.com/nomic-ai/gpt4all-chat  
❤️ llama.cpp: https://github.com/ggerganov/llama.cpp  
❤️ oobabooga text generation webui: https://github.com/oobabooga/text-generation-webui  
❤️ Colab webui inspired by camenduru: https://github.com/camenduru/text-generation-webui-colab/tree/main  
❤️ The Bloke for quantization of the models: https://huggingface.co/TheBloke  

## Coding models tested & average scores:
(All scores are from GPT-4-0613.)

| Model_Name                                                           | Avg_Scores | Colab_Link                                                                                                                                                                                                                                             | Date_Added | Link                                                              |
| :------------------------------------------------------------------- | :--------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------- | :---------------------------------------------------------------- |
| CodeLlama-13B-oasst-sft-v10 (using oobabooga/text-generation-webui)  | 9.8        | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/CodeLlama_13B_oasst_sft_v10.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | 2023/08/28 | https://huggingface.co/TheBloke/CodeLlama-13B-oasst-sft-v10-GGUF  |
| WizardCoder-Python-13B-V1.0 (using oobabooga/text-generation-webui)  | 9.5        | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/WizardCoder_Python_13B_V1_0.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | 2023/08/28 | https://huggingface.co/TheBloke/WizardCoder-Python-13B-V1.0-GGUF  |
| Redmond-Hermes-Coder-GPTQ (using oobabooga/text-generation-webui)    | 8.4        | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/Redmond_Hermes_Coder_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>            | 2023/07/03 | https://huggingface.co/TheBloke/Redmond-Hermes-Coder-GPTQ         |
| CodeUp-Alpha-13B-HF (using oobabooga/text-generation-webui)          | 7.9        | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/CodeUp_Alpha_13B_HF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>         | 2023/08/15 | https://huggingface.co/TheBloke/CodeUp-Alpha-13B-HF-GGML          |
| starchat-beta-GPTQ (using oobabooga/text-generation-webui)           | 7.6        | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/starchat_beta_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                   | 2023/07/04 | https://huggingface.co/TheBloke/starchat-beta-GPTQ                |
| wizard-vicuna-13B-GPTQ (using oobabooga/text-generation-webui)       | 7.3        | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/wizard_vicuna_13B_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>               | 2023/07/03 | https://huggingface.co/TheBloke/wizard-vicuna-13B-GPTQ            |
| WizardCoder-Guanaco-15B-V1.1 (using oobabooga/text-generation-webui) | 7.1        | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/WizardCoder_Guanaco_15B_V1_1.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>         | 2023/07/21 | https://huggingface.co/TheBloke/WizardCoder-Guanaco-15B-V1.1-GPTQ |
| CodeLlama-13B-Instruct (using oobabooga/text-generation-webui)       | 7          | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/CodeLlama_13B_Instruct.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>      | 2023/08/28 | https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF   |
| CodeUp-Llama-2-13B-Chat-HF (using oobabooga/text-generation-webui)   | 6          | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-Comparison-Colab-UI/blob/main/CodeUp_Llama_2_13B_Chat_HF.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>  | 2023/08/03 | https://huggingface.co/TheBloke/CodeUp-Llama-2-13B-Chat-HF-GGML   |
| WizardCoder-15B-1.0-GPTQ (using oobabooga/text-generation-webui)     | 5.9        | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/WizardCoder_15B_1_0_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>             | 2023/07/03 | https://huggingface.co/TheBloke/WizardCoder-15B-1.0-GPTQ          |
| wizardLM-13B-1.0-GPTQ (using oobabooga/text-generation-webui)        | 5.9        | <a target="_blank" href="https://colab.research.google.com/github/Troyanovsky/Local-LLM-comparison/blob/main/wizardLM_13B_1_0_GPTQ.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                | 2023/05/29 | https://huggingface.co/TheBloke/wizardLM-13B-1.0-GPTQ             |

## Questions and scores
Original responses can be found at: https://docs.google.com/spreadsheets/d/1ogDXUiaBx3t7EpMo44aaA6U6kLXX0x2tGRgLg8CISGs/edit?usp=sharing

### Question 1: Translate the following English text into French: "The sun rises in the east and sets in the west."  
Task Domain: Translation  
Expected Good Response: "Le soleil se lève à l'est et se couche à l'ouest."  
Explanation: This task tests the model's ability to understand and accurately translate text between languages.  
- wizardLM-7B.q4_2 (in GPT4All) : 10  
- gpt4all-j-v1.3-groovy (in GPT4All) : 1  
- vicuna-13b-1.1-q4_2 (in GPT4All) : 8  
- gpt4-x-alpaca-13b-ggml-q4_0 (using llama.cpp) : 10  
- koala-13B-4bit-128g.GGML (using llama.cpp) : 7  
- wizard-vicuna-13B.ggml.q4_0 (using llama.cpp) : 10  
- mpt-7b-chat (in GPT4All) : 1  
- stable-vicuna-13B-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 1  
- mpt-7b-instruct : 1  
- wizard-lm-uncensored-13b-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 7  
- Manticore-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- manticore_13b_chat_pyg_GPTQ (using oobabooga/text-generation-webui) : 10  
- Project-Baize-v2-13B-GPTQ (using oobabooga/text-generation-webui) : 8  
- Airoboros-13B-GPTQ-4bit (using oobabooga/text-generation-webui) : 10  
- guanaco-13B-GPTQ (using oobabooga/text-generation-webui) : 1  
- wizardLM-13B-1.0-GPTQ (using oobabooga/text-generation-webui) : 8  
- falcon-7b-instruct-GPTQ  (using oobabooga/text-generation-webui) : 10  
- WizardLM-Uncensored-Falcon-7B-GPTQ   (using oobabooga/text-generation-webui) : 8  
- Nous-Hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- 13B-HyperMantis_GPTQ (using oobabooga/text-generation-webui) : 1  
- Selfee-13B-GPTQ (using oobabooga/text-generation-webui) : 7  
- minotaur-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- Vicuna-13B-CoT-GPTQ (using oobabooga/text-generation-webui) : 8  
- CAMEL_13B_Combined_Data_GPTQ (using oobabooga/text-generation-webui) : 10  
- tulu-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- chronos-hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 1  
- robin-13B-v2-GPTQ (using oobabooga/text-generation-webui) : 10  
- minotaur-15B-GPTQ (using oobabooga/text-generation-webui) : 1  
- orca-mini-13b.ggmlv3.q5_K_M (using oobabooga/text-generation-webui) : 1  
- vicuna-13b-v1.3.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- vicuna-7B-v1.3-GPTQ (using oobabooga/text-generation-webui) : 8  
- UltraLM-13B-GPTQ (using oobabooga/text-generation-webui) : 7  
- airoboros-13B-gpt4-1.4-GPTQ (using oobabooga/text-generation-webui) : 8  
- llama-13b-supercot-GGML (using oobabooga/text-generation-webui) : 10  
- baichuan-vicuna-7B-GGML (using oobabooga/text-generation-webui) : 1  
- WizardLM-13B-V1.1 (using oobabooga/text-generation-webui) : 8.5  
- Llama-2-13B-chat (using oobabooga/text-generation-webui) : 10  
- Luna-AI-Llama2-Uncensored (using oobabooga/text-generation-webui) : 6  
- MythoLogic-13B (using oobabooga/text-generation-webui) : 10  
- Redmond-Puffin-13B (using oobabooga/text-generation-webui) : 10  
- llama-2-13B-Guanaco-QLoRA (using oobabooga/text-generation-webui) : 7.5  
- Dolphin-Llama-13B (using oobabooga/text-generation-webui) : 10  
- Nous-Hermes-Llama2 (using oobabooga/text-generation-webui) : 10  
- 13B-BlueMethod (using oobabooga/text-generation-webui) : 4  
- MythoBoros-13B (using oobabooga/text-generation-webui) : 10  
- WizardLM-13B-V1.2 (using oobabooga/text-generation-webui) : 10  
- 13B-Ouroboros (using oobabooga/text-generation-webui) : 10  
- AlpacaCielo-13B (using oobabooga/text-generation-webui) : 9  
- OpenChat_v3.2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-2.0 (using oobabooga/text-generation-webui) : 10  
- StableBeluga-13B (using oobabooga/text-generation-webui) : 10  
- Kimiko-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-13B-v2 (using oobabooga/text-generation-webui) : 10  
- WizardLM-1.0-Uncensored-Llama2-13B (using oobabooga/text-generation-webui) : 10  
- Huginn-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-Beluga-v2-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-Hermes-13B-v2 (using oobabooga/text-generation-webui) : 10  
- vicuna-13B-v1.5 (using oobabooga/text-generation-webui) : 10  
- Stable-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- Camel-Platypus2-13B (using oobabooga/text-generation-webui) : 7.5  
- OpenOrca-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- PuddleJumper-13B (using oobabooga/text-generation-webui) : 10  
- Kimiko-v2-13B (using oobabooga/text-generation-webui) : 7.5  
- Mythical-Destroyer-V2-L2-13B (using oobabooga/text-generation-webui) : 10  
- Athena-v1 (using oobabooga/text-generation-webui) : 10  
- Luban-13B (using oobabooga/text-generation-webui) : 10  
- OpenBuddy-Llama2-13B-v11.1 (using oobabooga/text-generation-webui) : 9.5  
- Kuchiki-1.1-L2-7B (using oobabooga/text-generation-webui) : 9.5  
- Pygmalion-2-13B-SuperCOT2 (using oobabooga/text-generation-webui) : 9.5  
- airoboros-l2-13b-gpt4-m2.0 (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-OpenOrca (using oobabooga/text-generation-webui) : 10  
- Nous-Capybara-7B (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-Instruct-v0.1 (using oobabooga/text-generation-webui) : 9.5  

### Question 2: Summarize the following text: "The water cycle is a natural process that involves the continuous movement of water on, above, and below the Earth's surface. It includes various stages like evaporation, condensation, precipitation, and runoff. This cycle plays a crucial role in maintaining Earth's water balance and supporting life."  
Task Domain: Summary  
Expected Good Response: "The water cycle is the continuous movement of water on Earth, crucial for maintaining water balance and supporting life."  
Explanation: This task evaluates the model's ability to extract the main points from a given text and generate a concise summary.  
- wizardLM-7B.q4_2 (in GPT4All) : 5  
- gpt4all-j-v1.3-groovy (in GPT4All) : 7.5  
- vicuna-13b-1.1-q4_2 (in GPT4All) : 9  
- gpt4-x-alpaca-13b-ggml-q4_0 (using llama.cpp) : 4  
- koala-13B-4bit-128g.GGML (using llama.cpp) : 10  
- wizard-vicuna-13B.ggml.q4_0 (using llama.cpp) : 10  
- mpt-7b-chat (in GPT4All) : 10  
- stable-vicuna-13B-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 10  
- mpt-7b-instruct : 7.5  
- wizard-lm-uncensored-13b-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 1  
- Manticore-13B-GPTQ (using oobabooga/text-generation-webui) : 9  
- manticore_13b_chat_pyg_GPTQ (using oobabooga/text-generation-webui) : 8.5  
- Project-Baize-v2-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- Airoboros-13B-GPTQ-4bit (using oobabooga/text-generation-webui) : 7.5  
- guanaco-13B-GPTQ (using oobabooga/text-generation-webui) : 9  
- wizardLM-13B-1.0-GPTQ (using oobabooga/text-generation-webui) : 9  
- falcon-7b-instruct-GPTQ  (using oobabooga/text-generation-webui) : 9  
- WizardLM-Uncensored-Falcon-7B-GPTQ   (using oobabooga/text-generation-webui) : 5  
- Nous-Hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- 13B-HyperMantis_GPTQ (using oobabooga/text-generation-webui) : 10  
- Selfee-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- minotaur-13B-GPTQ (using oobabooga/text-generation-webui) : 9  
- Vicuna-13B-CoT-GPTQ (using oobabooga/text-generation-webui) : 9  
- CAMEL_13B_Combined_Data_GPTQ (using oobabooga/text-generation-webui) : 9  
- tulu-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- chronos-hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- robin-13B-v2-GPTQ (using oobabooga/text-generation-webui) : 8.5  
- minotaur-15B-GPTQ (using oobabooga/text-generation-webui) : 8  
- orca-mini-13b.ggmlv3.q5_K_M (using oobabooga/text-generation-webui) : 1  
- vicuna-13b-v1.3.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- vicuna-7B-v1.3-GPTQ (using oobabooga/text-generation-webui) : 10  
- UltraLM-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- airoboros-13B-gpt4-1.4-GPTQ (using oobabooga/text-generation-webui) : 9  
- llama-13b-supercot-GGML (using oobabooga/text-generation-webui) : 10  
- baichuan-vicuna-7B-GGML (using oobabooga/text-generation-webui) : 10  
- WizardLM-13B-V1.1 (using oobabooga/text-generation-webui) : 10  
- Llama-2-13B-chat (using oobabooga/text-generation-webui) : 9  
- Luna-AI-Llama2-Uncensored (using oobabooga/text-generation-webui) : 8  
- MythoLogic-13B (using oobabooga/text-generation-webui) : 7.5  
- Redmond-Puffin-13B (using oobabooga/text-generation-webui) : 1  
- llama-2-13B-Guanaco-QLoRA (using oobabooga/text-generation-webui) : 10  
- Dolphin-Llama-13B (using oobabooga/text-generation-webui) : 10  
- Nous-Hermes-Llama2 (using oobabooga/text-generation-webui) : 10  
- 13B-BlueMethod (using oobabooga/text-generation-webui) : 1  
- MythoBoros-13B (using oobabooga/text-generation-webui) : 7  
- WizardLM-13B-V1.2 (using oobabooga/text-generation-webui) : 10  
- 13B-Ouroboros (using oobabooga/text-generation-webui) : 7.5  
- AlpacaCielo-13B (using oobabooga/text-generation-webui) : 9  
- OpenChat_v3.2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-2.0 (using oobabooga/text-generation-webui) : 7.5  
- StableBeluga-13B (using oobabooga/text-generation-webui) : 10  
- Kimiko-13B (using oobabooga/text-generation-webui) : 7.5  
- Chronos-13B-v2 (using oobabooga/text-generation-webui) : 9  
- WizardLM-1.0-Uncensored-Llama2-13B (using oobabooga/text-generation-webui) : 5  
- Huginn-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-Beluga-v2-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-Hermes-13B-v2 (using oobabooga/text-generation-webui) : 10  
- vicuna-13B-v1.5 (using oobabooga/text-generation-webui) : 10  
- Stable-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- Platypus2-13B (using oobabooga/text-generation-webui) : 7.5  
- Camel-Platypus2-13B (using oobabooga/text-generation-webui) : 9  
- OpenOrca-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- PuddleJumper-13B (using oobabooga/text-generation-webui) : 9.5  
- Kimiko-v2-13B (using oobabooga/text-generation-webui) : 9  
- Mythical-Destroyer-V2-L2-13B (using oobabooga/text-generation-webui) : 5  
- Athena-v1 (using oobabooga/text-generation-webui) : 9  
- Luban-13B (using oobabooga/text-generation-webui) : 10  
- OpenBuddy-Llama2-13B-v11.1 (using oobabooga/text-generation-webui) : 10  
- Kuchiki-1.1-L2-7B (using oobabooga/text-generation-webui) : 10  
- Pygmalion-2-13B-SuperCOT2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-m2.0 (using oobabooga/text-generation-webui) : 9.5  
- Mistral-7B-OpenOrca (using oobabooga/text-generation-webui) : 10  
- Nous-Capybara-7B (using oobabooga/text-generation-webui) : 9  
- Mistral-7B-Instruct-v0.1 (using oobabooga/text-generation-webui) : 10  

### Question 3: I want you to act as a senior software developer with deep knowledge in system design, frontend programming, and backend programming. Provide a high level design of a mental health journal app. Include the frontend and backend components. Do not write code.  
Task Domain: App Design  
Explanation: This task evaluates the model's ability to closely follow user's instruction for a complex task.  
- wizardLM-7B.q4_2 (in GPT4All) : 10  
- gpt4all-j-v1.3-groovy (in GPT4All) : 9  
- vicuna-13b-1.1-q4_2 (in GPT4All) : 10  
- gpt4-x-alpaca-13b-ggml-q4_0 (using llama.cpp) : 7.5  
- koala-13B-4bit-128g.GGML (using llama.cpp) : 10  
- wizard-vicuna-13B.ggml.q4_0 (using llama.cpp) : 10  
- mpt-7b-chat (in GPT4All) : 10  
- stable-vicuna-13B-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 10  
- mpt-7b-instruct : 6.5  
- wizard-lm-uncensored-13b-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 10  
- Manticore-13B-GPTQ (using oobabooga/text-generation-webui) : 8  
- manticore_13b_chat_pyg_GPTQ (using oobabooga/text-generation-webui) : 10  
- Project-Baize-v2-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- Airoboros-13B-GPTQ-4bit (using oobabooga/text-generation-webui) : 10  
- guanaco-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- wizardLM-13B-1.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- falcon-7b-instruct-GPTQ  (using oobabooga/text-generation-webui) : 9  
- WizardLM-Uncensored-Falcon-7B-GPTQ   (using oobabooga/text-generation-webui) : 10  
- Nous-Hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- 13B-HyperMantis_GPTQ (using oobabooga/text-generation-webui) : 10  
- Selfee-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- minotaur-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- Vicuna-13B-CoT-GPTQ (using oobabooga/text-generation-webui) : 10  
- CAMEL_13B_Combined_Data_GPTQ (using oobabooga/text-generation-webui) : 10  
- tulu-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- chronos-hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- robin-13B-v2-GPTQ (using oobabooga/text-generation-webui) : 10  
- minotaur-15B-GPTQ (using oobabooga/text-generation-webui) : 10  
- orca-mini-13b.ggmlv3.q5_K_M (using oobabooga/text-generation-webui) : 9  
- vicuna-13b-v1.3.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- vicuna-7B-v1.3-GPTQ (using oobabooga/text-generation-webui) : 10  
- UltraLM-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- airoboros-13B-gpt4-1.4-GPTQ (using oobabooga/text-generation-webui) : 10  
- llama-13b-supercot-GGML (using oobabooga/text-generation-webui) : 9.5  
- baichuan-vicuna-7B-GGML (using oobabooga/text-generation-webui) : 3  
- WizardLM-13B-V1.1 (using oobabooga/text-generation-webui) : 10  
- Llama-2-13B-chat (using oobabooga/text-generation-webui) : 10  
- Luna-AI-Llama2-Uncensored (using oobabooga/text-generation-webui) : 10  
- MythoLogic-13B (using oobabooga/text-generation-webui) : 10  
- Redmond-Puffin-13B (using oobabooga/text-generation-webui) : 9.5  
- llama-2-13B-Guanaco-QLoRA (using oobabooga/text-generation-webui) : 10  
- Dolphin-Llama-13B (using oobabooga/text-generation-webui) : 10  
- Nous-Hermes-Llama2 (using oobabooga/text-generation-webui) : 10  
- 13B-BlueMethod (using oobabooga/text-generation-webui) : 10  
- MythoBoros-13B (using oobabooga/text-generation-webui) : 10  
- WizardLM-13B-V1.2 (using oobabooga/text-generation-webui) : 10  
- 13B-Ouroboros (using oobabooga/text-generation-webui) : 10  
- AlpacaCielo-13B (using oobabooga/text-generation-webui) : 10  
- OpenChat_v3.2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-2.0 (using oobabooga/text-generation-webui) : 10  
- StableBeluga-13B (using oobabooga/text-generation-webui) : 10  
- Kimiko-13B (using oobabooga/text-generation-webui) : 8.5  
- Chronos-13B-v2 (using oobabooga/text-generation-webui) : 10  
- WizardLM-1.0-Uncensored-Llama2-13B (using oobabooga/text-generation-webui) : 10  
- Huginn-13B (using oobabooga/text-generation-webui) : 7.4  
- Chronos-Beluga-v2-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-Hermes-13B-v2 (using oobabooga/text-generation-webui) : 10  
- vicuna-13B-v1.5 (using oobabooga/text-generation-webui) : 10  
- Stable-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- Camel-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- OpenOrca-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- PuddleJumper-13B (using oobabooga/text-generation-webui) : 10  
- Kimiko-v2-13B (using oobabooga/text-generation-webui) : 10  
- Mythical-Destroyer-V2-L2-13B (using oobabooga/text-generation-webui) : 10  
- Athena-v1 (using oobabooga/text-generation-webui) : 10  
- Luban-13B (using oobabooga/text-generation-webui) : 10  
- OpenBuddy-Llama2-13B-v11.1 (using oobabooga/text-generation-webui) : 10  
- Kuchiki-1.1-L2-7B (using oobabooga/text-generation-webui) : 10  
- Pygmalion-2-13B-SuperCOT2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-m2.0 (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-OpenOrca (using oobabooga/text-generation-webui) : 10  
- Nous-Capybara-7B (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-Instruct-v0.1 (using oobabooga/text-generation-webui) : 10  

### Question 4: What are the main causes of the French Revolution according to this passage: "The French Revolution, which took place between 1789 and 1799, was rooted in financial crises, social inequality, and Enlightenment ideas."  
Task Domain: Abstractive Question Answering  
Expected Good Response: Financial crises, social inequality, and Enlightenment ideas.  
Explanation: This task tests the model's ability to understand the context and generate an answer in its own words.  
- wizardLM-7B.q4_2 (in GPT4All) : 10  
- gpt4all-j-v1.3-groovy (in GPT4All) : 10  
- vicuna-13b-1.1-q4_2 (in GPT4All) : 10  
- gpt4-x-alpaca-13b-ggml-q4_0 (using llama.cpp) : 1  
- koala-13B-4bit-128g.GGML (using llama.cpp) : 10  
- wizard-vicuna-13B.ggml.q4_0 (using llama.cpp) : 10  
- mpt-7b-chat (in GPT4All) : 10  
- stable-vicuna-13B-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 10  
- mpt-7b-instruct : 6  
- wizard-lm-uncensored-13b-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 10  
- Manticore-13B-GPTQ (using oobabooga/text-generation-webui) : 4  
- manticore_13b_chat_pyg_GPTQ (using oobabooga/text-generation-webui) : 10  
- Project-Baize-v2-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- Airoboros-13B-GPTQ-4bit (using oobabooga/text-generation-webui) : 10  
- guanaco-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- wizardLM-13B-1.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- falcon-7b-instruct-GPTQ  (using oobabooga/text-generation-webui) : 10  
- WizardLM-Uncensored-Falcon-7B-GPTQ   (using oobabooga/text-generation-webui) : 10  
- Nous-Hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- 13B-HyperMantis_GPTQ (using oobabooga/text-generation-webui) : 10  
- Selfee-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- minotaur-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- Vicuna-13B-CoT-GPTQ (using oobabooga/text-generation-webui) : 10  
- CAMEL_13B_Combined_Data_GPTQ (using oobabooga/text-generation-webui) : 9.5  
- tulu-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- chronos-hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- robin-13B-v2-GPTQ (using oobabooga/text-generation-webui) : 10  
- minotaur-15B-GPTQ (using oobabooga/text-generation-webui) : 9  
- orca-mini-13b.ggmlv3.q5_K_M (using oobabooga/text-generation-webui) : 10  
- vicuna-13b-v1.3.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- vicuna-7B-v1.3-GPTQ (using oobabooga/text-generation-webui) : 10  
- UltraLM-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- airoboros-13B-gpt4-1.4-GPTQ (using oobabooga/text-generation-webui) : 10  
- llama-13b-supercot-GGML (using oobabooga/text-generation-webui) : 9.5  
- baichuan-vicuna-7B-GGML (using oobabooga/text-generation-webui) : 7.5  
- WizardLM-13B-V1.1 (using oobabooga/text-generation-webui) : 10  
- Llama-2-13B-chat (using oobabooga/text-generation-webui) : 10  
- Luna-AI-Llama2-Uncensored (using oobabooga/text-generation-webui) : 10  
- MythoLogic-13B (using oobabooga/text-generation-webui) : 10  
- Redmond-Puffin-13B (using oobabooga/text-generation-webui) : 10  
- llama-2-13B-Guanaco-QLoRA (using oobabooga/text-generation-webui) : 8  
- Dolphin-Llama-13B (using oobabooga/text-generation-webui) : 10  
- Nous-Hermes-Llama2 (using oobabooga/text-generation-webui) : 10  
- 13B-BlueMethod (using oobabooga/text-generation-webui) : 10  
- MythoBoros-13B (using oobabooga/text-generation-webui) : 10  
- WizardLM-13B-V1.2 (using oobabooga/text-generation-webui) : 10  
- 13B-Ouroboros (using oobabooga/text-generation-webui) : 10  
- AlpacaCielo-13B (using oobabooga/text-generation-webui) : 10  
- OpenChat_v3.2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-2.0 (using oobabooga/text-generation-webui) : 10  
- StableBeluga-13B (using oobabooga/text-generation-webui) : 10  
- Kimiko-13B (using oobabooga/text-generation-webui) : 9  
- Chronos-13B-v2 (using oobabooga/text-generation-webui) : 10  
- WizardLM-1.0-Uncensored-Llama2-13B (using oobabooga/text-generation-webui) : 10  
- Huginn-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-Beluga-v2-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-Hermes-13B-v2 (using oobabooga/text-generation-webui) : 10  
- vicuna-13B-v1.5 (using oobabooga/text-generation-webui) : 10  
- Stable-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- Camel-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- OpenOrca-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- PuddleJumper-13B (using oobabooga/text-generation-webui) : 10  
- Kimiko-v2-13B (using oobabooga/text-generation-webui) : 10  
- Mythical-Destroyer-V2-L2-13B (using oobabooga/text-generation-webui) : 10  
- Athena-v1 (using oobabooga/text-generation-webui) : 10  
- Luban-13B (using oobabooga/text-generation-webui) : 10  
- OpenBuddy-Llama2-13B-v11.1 (using oobabooga/text-generation-webui) : 10  
- Kuchiki-1.1-L2-7B (using oobabooga/text-generation-webui) : 10  
- Pygmalion-2-13B-SuperCOT2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-m2.0 (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-OpenOrca (using oobabooga/text-generation-webui) : 10  
- Nous-Capybara-7B (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-Instruct-v0.1 (using oobabooga/text-generation-webui) : 10  

### Question 5: In the following text, identify the two artists mentioned: "During the Renaissance, Leonardo da Vinci and Michelangelo were two of the most influential artists who created masterpieces that continue to inspire people today."  
Task Domain: Extractive Question Answering  
Expected Good Response: Leonardo da Vinci, Michelangelo  
Explanation: This task assesses the model's ability to extract specific information from a given text.  
- wizardLM-7B.q4_2 (in GPT4All) : 10  
- gpt4all-j-v1.3-groovy (in GPT4All) : 10  
- vicuna-13b-1.1-q4_2 (in GPT4All) : 10  
- gpt4-x-alpaca-13b-ggml-q4_0 (using llama.cpp) : 10  
- koala-13B-4bit-128g.GGML (using llama.cpp) : 10  
- wizard-vicuna-13B.ggml.q4_0 (using llama.cpp) : 10  
- mpt-7b-chat (in GPT4All) : 10  
- stable-vicuna-13B-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 10  
- mpt-7b-instruct : 9  
- wizard-lm-uncensored-13b-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 10  
- Manticore-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- manticore_13b_chat_pyg_GPTQ (using oobabooga/text-generation-webui) : 10  
- Project-Baize-v2-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- Airoboros-13B-GPTQ-4bit (using oobabooga/text-generation-webui) : 10  
- guanaco-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- wizardLM-13B-1.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- falcon-7b-instruct-GPTQ  (using oobabooga/text-generation-webui) : 10  
- WizardLM-Uncensored-Falcon-7B-GPTQ   (using oobabooga/text-generation-webui) : 10  
- Nous-Hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- 13B-HyperMantis_GPTQ (using oobabooga/text-generation-webui) : 10  
- Selfee-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- minotaur-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- Vicuna-13B-CoT-GPTQ (using oobabooga/text-generation-webui) : 10  
- CAMEL_13B_Combined_Data_GPTQ (using oobabooga/text-generation-webui) : 10  
- tulu-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- chronos-hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- robin-13B-v2-GPTQ (using oobabooga/text-generation-webui) : 10  
- minotaur-15B-GPTQ (using oobabooga/text-generation-webui) : 8.5  
- orca-mini-13b.ggmlv3.q5_K_M (using oobabooga/text-generation-webui) : 10  
- vicuna-13b-v1.3.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- vicuna-7B-v1.3-GPTQ (using oobabooga/text-generation-webui) : 10  
- UltraLM-13B-GPTQ (using oobabooga/text-generation-webui) : 8.5  
- airoboros-13B-gpt4-1.4-GPTQ (using oobabooga/text-generation-webui) : 10  
- llama-13b-supercot-GGML (using oobabooga/text-generation-webui) : 9.5  
- baichuan-vicuna-7B-GGML (using oobabooga/text-generation-webui) : 8.5  
- WizardLM-13B-V1.1 (using oobabooga/text-generation-webui) : 10  
- Llama-2-13B-chat (using oobabooga/text-generation-webui) : 10  
- Luna-AI-Llama2-Uncensored (using oobabooga/text-generation-webui) : 10  
- MythoLogic-13B (using oobabooga/text-generation-webui) : 10  
- Redmond-Puffin-13B (using oobabooga/text-generation-webui) : 10  
- llama-2-13B-Guanaco-QLoRA (using oobabooga/text-generation-webui) : 10  
- Dolphin-Llama-13B (using oobabooga/text-generation-webui) : 10  
- Nous-Hermes-Llama2 (using oobabooga/text-generation-webui) : 10  
- 13B-BlueMethod (using oobabooga/text-generation-webui) : 10  
- MythoBoros-13B (using oobabooga/text-generation-webui) : 10  
- WizardLM-13B-V1.2 (using oobabooga/text-generation-webui) : 10  
- 13B-Ouroboros (using oobabooga/text-generation-webui) : 10  
- AlpacaCielo-13B (using oobabooga/text-generation-webui) : 10  
- OpenChat_v3.2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-2.0 (using oobabooga/text-generation-webui) : 10  
- StableBeluga-13B (using oobabooga/text-generation-webui) : 10  
- Kimiko-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-13B-v2 (using oobabooga/text-generation-webui) : 10  
- WizardLM-1.0-Uncensored-Llama2-13B (using oobabooga/text-generation-webui) : 10  
- Huginn-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-Beluga-v2-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-Hermes-13B-v2 (using oobabooga/text-generation-webui) : 10  
- vicuna-13B-v1.5 (using oobabooga/text-generation-webui) : 10  
- Stable-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- Camel-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- OpenOrca-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- PuddleJumper-13B (using oobabooga/text-generation-webui) : 10  
- Kimiko-v2-13B (using oobabooga/text-generation-webui) : 10  
- Mythical-Destroyer-V2-L2-13B (using oobabooga/text-generation-webui) : 10  
- Athena-v1 (using oobabooga/text-generation-webui) : 10  
- Luban-13B (using oobabooga/text-generation-webui) : 10  
- OpenBuddy-Llama2-13B-v11.1 (using oobabooga/text-generation-webui) : 10  
- Kuchiki-1.1-L2-7B (using oobabooga/text-generation-webui) : 10  
- Pygmalion-2-13B-SuperCOT2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-m2.0 (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-OpenOrca (using oobabooga/text-generation-webui) : 10  
- Nous-Capybara-7B (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-Instruct-v0.1 (using oobabooga/text-generation-webui) : 10  

### Question 6: Determine the sentiment of this customer review: "I had a fantastic experience at this restaurant. The food was delicious, and the service was outstanding."  
Task Domain: Sentiment Analysis  
Expected Good Response: Positive  
Explanation: This task evaluates the model's ability to analyze text and identify the sentiment expressed.  
- wizardLM-7B.q4_2 (in GPT4All) : 10  
- gpt4all-j-v1.3-groovy (in GPT4All) : 10  
- vicuna-13b-1.1-q4_2 (in GPT4All) : 10  
- gpt4-x-alpaca-13b-ggml-q4_0 (using llama.cpp) : 10  
- koala-13B-4bit-128g.GGML (using llama.cpp) : 10  
- wizard-vicuna-13B.ggml.q4_0 (using llama.cpp) : 10  
- mpt-7b-chat (in GPT4All) : 10  
- stable-vicuna-13B-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 10  
- mpt-7b-instruct : 10  
- wizard-lm-uncensored-13b-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 10  
- Manticore-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- manticore_13b_chat_pyg_GPTQ (using oobabooga/text-generation-webui) : 10  
- Project-Baize-v2-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- Airoboros-13B-GPTQ-4bit (using oobabooga/text-generation-webui) : 10  
- guanaco-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- wizardLM-13B-1.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- falcon-7b-instruct-GPTQ  (using oobabooga/text-generation-webui) : 1  
- WizardLM-Uncensored-Falcon-7B-GPTQ   (using oobabooga/text-generation-webui) : 10  
- Nous-Hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- 13B-HyperMantis_GPTQ (using oobabooga/text-generation-webui) : 10  
- Selfee-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- minotaur-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- Vicuna-13B-CoT-GPTQ (using oobabooga/text-generation-webui) : 10  
- CAMEL_13B_Combined_Data_GPTQ (using oobabooga/text-generation-webui) : 4  
- tulu-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- chronos-hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- robin-13B-v2-GPTQ (using oobabooga/text-generation-webui) : 10  
- minotaur-15B-GPTQ (using oobabooga/text-generation-webui) : 1  
- orca-mini-13b.ggmlv3.q5_K_M (using oobabooga/text-generation-webui) : 8.5  
- vicuna-13b-v1.3.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- vicuna-7B-v1.3-GPTQ (using oobabooga/text-generation-webui) : 10  
- UltraLM-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- airoboros-13B-gpt4-1.4-GPTQ (using oobabooga/text-generation-webui) : 10  
- llama-13b-supercot-GGML (using oobabooga/text-generation-webui) : 10  
- baichuan-vicuna-7B-GGML (using oobabooga/text-generation-webui) : 5  
- WizardLM-13B-V1.1 (using oobabooga/text-generation-webui) : 10  
- Llama-2-13B-chat (using oobabooga/text-generation-webui) : 10  
- Luna-AI-Llama2-Uncensored (using oobabooga/text-generation-webui) : 10  
- MythoLogic-13B (using oobabooga/text-generation-webui) : 10  
- Redmond-Puffin-13B (using oobabooga/text-generation-webui) : 10  
- llama-2-13B-Guanaco-QLoRA (using oobabooga/text-generation-webui) : 10  
- Dolphin-Llama-13B (using oobabooga/text-generation-webui) : 10  
- Nous-Hermes-Llama2 (using oobabooga/text-generation-webui) : 10  
- 13B-BlueMethod (using oobabooga/text-generation-webui) : 10  
- MythoBoros-13B (using oobabooga/text-generation-webui) : 10  
- WizardLM-13B-V1.2 (using oobabooga/text-generation-webui) : 10  
- 13B-Ouroboros (using oobabooga/text-generation-webui) : 10  
- AlpacaCielo-13B (using oobabooga/text-generation-webui) : 10  
- OpenChat_v3.2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-2.0 (using oobabooga/text-generation-webui) : 10  
- StableBeluga-13B (using oobabooga/text-generation-webui) : 10  
- Kimiko-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-13B-v2 (using oobabooga/text-generation-webui) : 10  
- WizardLM-1.0-Uncensored-Llama2-13B (using oobabooga/text-generation-webui) : 10  
- Huginn-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-Beluga-v2-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-Hermes-13B-v2 (using oobabooga/text-generation-webui) : 10  
- vicuna-13B-v1.5 (using oobabooga/text-generation-webui) : 10  
- Stable-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- Camel-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- OpenOrca-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- PuddleJumper-13B (using oobabooga/text-generation-webui) : 10  
- Kimiko-v2-13B (using oobabooga/text-generation-webui) : 10  
- Mythical-Destroyer-V2-L2-13B (using oobabooga/text-generation-webui) : 10  
- Athena-v1 (using oobabooga/text-generation-webui) : 10  
- Luban-13B (using oobabooga/text-generation-webui) : 10  
- OpenBuddy-Llama2-13B-v11.1 (using oobabooga/text-generation-webui) : 10  
- Kuchiki-1.1-L2-7B (using oobabooga/text-generation-webui) : 10  
- Pygmalion-2-13B-SuperCOT2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-m2.0 (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-OpenOrca (using oobabooga/text-generation-webui) : 10  
- Nous-Capybara-7B (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-Instruct-v0.1 (using oobabooga/text-generation-webui) : 10  

### Question 7: Classify the following text into one of these categories: Sports, Technology, Health, or Politics: "Apple recently unveiled its latest iPhone, featuring a faster processor and improved battery life."  
Task Domain: Zero-shot Classification  
Expected Good Response: Technology  
Explanation: This task tests the model's ability to classify text into predefined categories without any prior training on the specific categories.  
- wizardLM-7B.q4_2 (in GPT4All) : 10  
- gpt4all-j-v1.3-groovy (in GPT4All) : 5  
- vicuna-13b-1.1-q4_2 (in GPT4All) : 1  
- gpt4-x-alpaca-13b-ggml-q4_0 (using llama.cpp) : 10  
- koala-13B-4bit-128g.GGML (using llama.cpp) : 10  
- wizard-vicuna-13B.ggml.q4_0 (using llama.cpp) : 10  
- mpt-7b-chat (in GPT4All) : 10  
- stable-vicuna-13B-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 10  
- mpt-7b-instruct : 10  
- wizard-lm-uncensored-13b-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 10  
- Manticore-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- manticore_13b_chat_pyg_GPTQ (using oobabooga/text-generation-webui) : 10  
- Project-Baize-v2-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- Airoboros-13B-GPTQ-4bit (using oobabooga/text-generation-webui) : 10  
- guanaco-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- wizardLM-13B-1.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- falcon-7b-instruct-GPTQ  (using oobabooga/text-generation-webui) : 10  
- WizardLM-Uncensored-Falcon-7B-GPTQ   (using oobabooga/text-generation-webui) : 10  
- Nous-Hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- 13B-HyperMantis_GPTQ (using oobabooga/text-generation-webui) : 10  
- Selfee-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- minotaur-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- Vicuna-13B-CoT-GPTQ (using oobabooga/text-generation-webui) : 10  
- CAMEL_13B_Combined_Data_GPTQ (using oobabooga/text-generation-webui) : 10  
- tulu-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- chronos-hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- robin-13B-v2-GPTQ (using oobabooga/text-generation-webui) : 10  
- minotaur-15B-GPTQ (using oobabooga/text-generation-webui) : 10  
- orca-mini-13b.ggmlv3.q5_K_M (using oobabooga/text-generation-webui) : 10  
- vicuna-13b-v1.3.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- vicuna-7B-v1.3-GPTQ (using oobabooga/text-generation-webui) : 10  
- UltraLM-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- airoboros-13B-gpt4-1.4-GPTQ (using oobabooga/text-generation-webui) : 10  
- llama-13b-supercot-GGML (using oobabooga/text-generation-webui) : 10  
- baichuan-vicuna-7B-GGML (using oobabooga/text-generation-webui) : 10  
- WizardLM-13B-V1.1 (using oobabooga/text-generation-webui) : 10  
- Llama-2-13B-chat (using oobabooga/text-generation-webui) : 10  
- Luna-AI-Llama2-Uncensored (using oobabooga/text-generation-webui) : 10  
- MythoLogic-13B (using oobabooga/text-generation-webui) : 10  
- Redmond-Puffin-13B (using oobabooga/text-generation-webui) : 2  
- llama-2-13B-Guanaco-QLoRA (using oobabooga/text-generation-webui) : 10  
- Dolphin-Llama-13B (using oobabooga/text-generation-webui) : 10  
- Nous-Hermes-Llama2 (using oobabooga/text-generation-webui) : 10  
- 13B-BlueMethod (using oobabooga/text-generation-webui) : 10  
- MythoBoros-13B (using oobabooga/text-generation-webui) : 10  
- WizardLM-13B-V1.2 (using oobabooga/text-generation-webui) : 10  
- 13B-Ouroboros (using oobabooga/text-generation-webui) : 10  
- AlpacaCielo-13B (using oobabooga/text-generation-webui) : 10  
- OpenChat_v3.2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-2.0 (using oobabooga/text-generation-webui) : 10  
- StableBeluga-13B (using oobabooga/text-generation-webui) : 10  
- Kimiko-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-13B-v2 (using oobabooga/text-generation-webui) : 10  
- WizardLM-1.0-Uncensored-Llama2-13B (using oobabooga/text-generation-webui) : 10  
- Huginn-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-Beluga-v2-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-Hermes-13B-v2 (using oobabooga/text-generation-webui) : 10  
- vicuna-13B-v1.5 (using oobabooga/text-generation-webui) : 10  
- Stable-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- Platypus2-13B (using oobabooga/text-generation-webui) : 4  
- Camel-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- OpenOrca-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- PuddleJumper-13B (using oobabooga/text-generation-webui) : 10  
- Kimiko-v2-13B (using oobabooga/text-generation-webui) : 10  
- Mythical-Destroyer-V2-L2-13B (using oobabooga/text-generation-webui) : 10  
- Athena-v1 (using oobabooga/text-generation-webui) : 10  
- Luban-13B (using oobabooga/text-generation-webui) : 10  
- OpenBuddy-Llama2-13B-v11.1 (using oobabooga/text-generation-webui) : 10  
- Kuchiki-1.1-L2-7B (using oobabooga/text-generation-webui) : 10  
- Pygmalion-2-13B-SuperCOT2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-m2.0 (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-OpenOrca (using oobabooga/text-generation-webui) : 10  
- Nous-Capybara-7B (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-Instruct-v0.1 (using oobabooga/text-generation-webui) : 10  

### Question 8: Complete the following sentence by filling in the blank: "The capital city of France is _______."  
Task Domain: Mask Filling  
Expected Good Response: Paris  
Explanation: This task assesses the model's ability to understand context and generate appropriate words to fill in missing information.  
- wizardLM-7B.q4_2 (in GPT4All) : 10  
- gpt4all-j-v1.3-groovy (in GPT4All) : 10  
- vicuna-13b-1.1-q4_2 (in GPT4All) : 10  
- gpt4-x-alpaca-13b-ggml-q4_0 (using llama.cpp) : 5  
- koala-13B-4bit-128g.GGML (using llama.cpp) : 10  
- wizard-vicuna-13B.ggml.q4_0 (using llama.cpp) : 10  
- mpt-7b-chat (in GPT4All) : 10  
- stable-vicuna-13B-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 10  
- mpt-7b-instruct : 10  
- wizard-lm-uncensored-13b-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 10  
- Manticore-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- manticore_13b_chat_pyg_GPTQ (using oobabooga/text-generation-webui) : 10  
- Project-Baize-v2-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- Airoboros-13B-GPTQ-4bit (using oobabooga/text-generation-webui) : 10  
- guanaco-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- wizardLM-13B-1.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- falcon-7b-instruct-GPTQ  (using oobabooga/text-generation-webui) : 10  
- WizardLM-Uncensored-Falcon-7B-GPTQ   (using oobabooga/text-generation-webui) : 10  
- Nous-Hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- 13B-HyperMantis_GPTQ (using oobabooga/text-generation-webui) : 10  
- Selfee-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- minotaur-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- Vicuna-13B-CoT-GPTQ (using oobabooga/text-generation-webui) : 10  
- CAMEL_13B_Combined_Data_GPTQ (using oobabooga/text-generation-webui) : 10  
- tulu-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- chronos-hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- robin-13B-v2-GPTQ (using oobabooga/text-generation-webui) : 10  
- minotaur-15B-GPTQ (using oobabooga/text-generation-webui) : 10  
- orca-mini-13b.ggmlv3.q5_K_M (using oobabooga/text-generation-webui) : 10  
- vicuna-13b-v1.3.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- vicuna-7B-v1.3-GPTQ (using oobabooga/text-generation-webui) : 10  
- UltraLM-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- airoboros-13B-gpt4-1.4-GPTQ (using oobabooga/text-generation-webui) : 10  
- llama-13b-supercot-GGML (using oobabooga/text-generation-webui) : 10  
- baichuan-vicuna-7B-GGML (using oobabooga/text-generation-webui) : 10  
- WizardLM-13B-V1.1 (using oobabooga/text-generation-webui) : 10  
- Llama-2-13B-chat (using oobabooga/text-generation-webui) : 10  
- Luna-AI-Llama2-Uncensored (using oobabooga/text-generation-webui) : 10  
- MythoLogic-13B (using oobabooga/text-generation-webui) : 10  
- Redmond-Puffin-13B (using oobabooga/text-generation-webui) : 10  
- llama-2-13B-Guanaco-QLoRA (using oobabooga/text-generation-webui) : 10  
- Dolphin-Llama-13B (using oobabooga/text-generation-webui) : 10  
- Nous-Hermes-Llama2 (using oobabooga/text-generation-webui) : 10  
- 13B-BlueMethod (using oobabooga/text-generation-webui) : 10  
- MythoBoros-13B (using oobabooga/text-generation-webui) : 10  
- WizardLM-13B-V1.2 (using oobabooga/text-generation-webui) : 10  
- 13B-Ouroboros (using oobabooga/text-generation-webui) : 10  
- AlpacaCielo-13B (using oobabooga/text-generation-webui) : 10  
- OpenChat_v3.2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-2.0 (using oobabooga/text-generation-webui) : 10  
- StableBeluga-13B (using oobabooga/text-generation-webui) : 10  
- Kimiko-13B (using oobabooga/text-generation-webui) : 1  
- Chronos-13B-v2 (using oobabooga/text-generation-webui) : 10  
- WizardLM-1.0-Uncensored-Llama2-13B (using oobabooga/text-generation-webui) : 10  
- Huginn-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-Beluga-v2-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-Hermes-13B-v2 (using oobabooga/text-generation-webui) : 10  
- vicuna-13B-v1.5 (using oobabooga/text-generation-webui) : 10  
- Stable-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- Camel-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- OpenOrca-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- PuddleJumper-13B (using oobabooga/text-generation-webui) : 10  
- Kimiko-v2-13B (using oobabooga/text-generation-webui) : 10  
- Mythical-Destroyer-V2-L2-13B (using oobabooga/text-generation-webui) : 5  
- Athena-v1 (using oobabooga/text-generation-webui) : 10  
- Luban-13B (using oobabooga/text-generation-webui) : 10  
- OpenBuddy-Llama2-13B-v11.1 (using oobabooga/text-generation-webui) : 10  
- Kuchiki-1.1-L2-7B (using oobabooga/text-generation-webui) : 10  
- Pygmalion-2-13B-SuperCOT2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-m2.0 (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-OpenOrca (using oobabooga/text-generation-webui) : 10  
- Nous-Capybara-7B (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-Instruct-v0.1 (using oobabooga/text-generation-webui) : 10  

### Question 9: Write a rhyming couplet about nature.  
Task Domain: Poetry Generation  
Expected Good Response: "In nature's beauty, we find respite and grace,
A symphony of colors that time cannot erase."  
Explanation: This task tests the model's ability to generate creative and coherent text that adheres to specific constraints, such as rhyme and theme.  
- wizardLM-7B.q4_2 (in GPT4All) : 10  
- gpt4all-j-v1.3-groovy (in GPT4All) : 7.5  
- vicuna-13b-1.1-q4_2 (in GPT4All) : 10  
- gpt4-x-alpaca-13b-ggml-q4_0 (using llama.cpp) : 1  
- koala-13B-4bit-128g.GGML (using llama.cpp) : 6.5  
- wizard-vicuna-13B.ggml.q4_0 (using llama.cpp) : 10  
- mpt-7b-chat (in GPT4All) : 10  
- stable-vicuna-13B-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 6  
- mpt-7b-instruct : 2  
- wizard-lm-uncensored-13b-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 10  
- Manticore-13B-GPTQ (using oobabooga/text-generation-webui) : 7.5  
- manticore_13b_chat_pyg_GPTQ (using oobabooga/text-generation-webui) : 8  
- Project-Baize-v2-13B-GPTQ (using oobabooga/text-generation-webui) : 9  
- Airoboros-13B-GPTQ-4bit (using oobabooga/text-generation-webui) : 10  
- guanaco-13B-GPTQ (using oobabooga/text-generation-webui) : 6.5  
- wizardLM-13B-1.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- falcon-7b-instruct-GPTQ  (using oobabooga/text-generation-webui) : 10  
- WizardLM-Uncensored-Falcon-7B-GPTQ   (using oobabooga/text-generation-webui) : 7.5  
- Nous-Hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 9  
- 13B-HyperMantis_GPTQ (using oobabooga/text-generation-webui) : 10  
- Selfee-13B-GPTQ (using oobabooga/text-generation-webui) : 7.5  
- minotaur-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- Vicuna-13B-CoT-GPTQ (using oobabooga/text-generation-webui) : 10  
- CAMEL_13B_Combined_Data_GPTQ (using oobabooga/text-generation-webui) : 4  
- tulu-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- chronos-hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 6  
- robin-13B-v2-GPTQ (using oobabooga/text-generation-webui) : 7  
- minotaur-15B-GPTQ (using oobabooga/text-generation-webui) : 7.5  
- orca-mini-13b.ggmlv3.q5_K_M (using oobabooga/text-generation-webui) : 7.5  
- vicuna-13b-v1.3.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- vicuna-7B-v1.3-GPTQ (using oobabooga/text-generation-webui) : 7  
- UltraLM-13B-GPTQ (using oobabooga/text-generation-webui) : 7.5  
- airoboros-13B-gpt4-1.4-GPTQ (using oobabooga/text-generation-webui) : 10  
- llama-13b-supercot-GGML (using oobabooga/text-generation-webui) : 4  
- baichuan-vicuna-7B-GGML (using oobabooga/text-generation-webui) : 4  
- WizardLM-13B-V1.1 (using oobabooga/text-generation-webui) : 10  
- Llama-2-13B-chat (using oobabooga/text-generation-webui) : 8  
- Luna-AI-Llama2-Uncensored (using oobabooga/text-generation-webui) : 9  
- MythoLogic-13B (using oobabooga/text-generation-webui) : 10  
- Redmond-Puffin-13B (using oobabooga/text-generation-webui) : 4  
- llama-2-13B-Guanaco-QLoRA (using oobabooga/text-generation-webui) : 7  
- Dolphin-Llama-13B (using oobabooga/text-generation-webui) : 7.5  
- Nous-Hermes-Llama2 (using oobabooga/text-generation-webui) : 10  
- 13B-BlueMethod (using oobabooga/text-generation-webui) : 4  
- MythoBoros-13B (using oobabooga/text-generation-webui) : 8  
- WizardLM-13B-V1.2 (using oobabooga/text-generation-webui) : 10  
- 13B-Ouroboros (using oobabooga/text-generation-webui) : 7  
- AlpacaCielo-13B (using oobabooga/text-generation-webui) : 10  
- OpenChat_v3.2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-2.0 (using oobabooga/text-generation-webui) : 8  
- StableBeluga-13B (using oobabooga/text-generation-webui) : 7.5  
- Kimiko-13B (using oobabooga/text-generation-webui) : 6.5  
- Chronos-13B-v2 (using oobabooga/text-generation-webui) : 10  
- WizardLM-1.0-Uncensored-Llama2-13B (using oobabooga/text-generation-webui) : 10  
- Huginn-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-Beluga-v2-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-Hermes-13B-v2 (using oobabooga/text-generation-webui) : 10  
- vicuna-13B-v1.5 (using oobabooga/text-generation-webui) : 10  
- Stable-Platypus2-13B (using oobabooga/text-generation-webui) : 8.5  
- Platypus2-13B (using oobabooga/text-generation-webui) : 6  
- Camel-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- OpenOrca-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- PuddleJumper-13B (using oobabooga/text-generation-webui) : 7.5  
- Kimiko-v2-13B (using oobabooga/text-generation-webui) : 4  
- Mythical-Destroyer-V2-L2-13B (using oobabooga/text-generation-webui) : 3  
- Athena-v1 (using oobabooga/text-generation-webui) : 10  
- Luban-13B (using oobabooga/text-generation-webui) : 7.4  
- OpenBuddy-Llama2-13B-v11.1 (using oobabooga/text-generation-webui) : 8  
- Kuchiki-1.1-L2-7B (using oobabooga/text-generation-webui) : 10  
- Pygmalion-2-13B-SuperCOT2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-m2.0 (using oobabooga/text-generation-webui) : 6.5  
- Mistral-7B-OpenOrca (using oobabooga/text-generation-webui) : 10  
- Nous-Capybara-7B (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-Instruct-v0.1 (using oobabooga/text-generation-webui) : 9  

### Question 10: Based on the following statement, determine if the author's opinion is for or against nuclear energy: "Nuclear energy is a powerful source, but the potential risks and radioactive waste management issues make it a dangerous choice."  
Task Domain: Opinion Detection  
Expected Good Response: Against  
Explanation: This task evaluates the model's ability to understand and identify the author's stance or opinion on a specific topic.  
- wizardLM-7B.q4_2 (in GPT4All) : 10  
- gpt4all-j-v1.3-groovy (in GPT4All) : 4  
- vicuna-13b-1.1-q4_2 (in GPT4All) : 10  
- gpt4-x-alpaca-13b-ggml-q4_0 (using llama.cpp) : 9.5  
- koala-13B-4bit-128g.GGML (using llama.cpp) : 6  
- wizard-vicuna-13B.ggml.q4_0 (using llama.cpp) : 10  
- mpt-7b-chat (in GPT4All) : 8.5  
- stable-vicuna-13B-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 10  
- mpt-7b-instruct : 10  
- wizard-lm-uncensored-13b-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 7.5  
- Manticore-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- manticore_13b_chat_pyg_GPTQ (using oobabooga/text-generation-webui) : 10  
- Project-Baize-v2-13B-GPTQ (using oobabooga/text-generation-webui) : 8.5  
- Airoboros-13B-GPTQ-4bit (using oobabooga/text-generation-webui) : 10  
- guanaco-13B-GPTQ (using oobabooga/text-generation-webui) : 9.5  
- wizardLM-13B-1.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- falcon-7b-instruct-GPTQ  (using oobabooga/text-generation-webui) : 10  
- WizardLM-Uncensored-Falcon-7B-GPTQ   (using oobabooga/text-generation-webui) : 10  
- Nous-Hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- 13B-HyperMantis_GPTQ (using oobabooga/text-generation-webui) : 10  
- Selfee-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- minotaur-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- Vicuna-13B-CoT-GPTQ (using oobabooga/text-generation-webui) : 10  
- CAMEL_13B_Combined_Data_GPTQ (using oobabooga/text-generation-webui) : 9.5  
- tulu-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- chronos-hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- robin-13B-v2-GPTQ (using oobabooga/text-generation-webui) : 8.5  
- minotaur-15B-GPTQ (using oobabooga/text-generation-webui) : 10  
- orca-mini-13b.ggmlv3.q5_K_M (using oobabooga/text-generation-webui) : 4  
- vicuna-13b-v1.3.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- vicuna-7B-v1.3-GPTQ (using oobabooga/text-generation-webui) : 4  
- UltraLM-13B-GPTQ (using oobabooga/text-generation-webui) : 6.5  
- airoboros-13B-gpt4-1.4-GPTQ (using oobabooga/text-generation-webui) : 10  
- llama-13b-supercot-GGML (using oobabooga/text-generation-webui) : 10  
- baichuan-vicuna-7B-GGML (using oobabooga/text-generation-webui) : 7  
- WizardLM-13B-V1.1 (using oobabooga/text-generation-webui) : 10  
- Llama-2-13B-chat (using oobabooga/text-generation-webui) : 10  
- Luna-AI-Llama2-Uncensored (using oobabooga/text-generation-webui) : 10  
- MythoLogic-13B (using oobabooga/text-generation-webui) : 10  
- Redmond-Puffin-13B (using oobabooga/text-generation-webui) : 10  
- llama-2-13B-Guanaco-QLoRA (using oobabooga/text-generation-webui) : 8.5  
- Dolphin-Llama-13B (using oobabooga/text-generation-webui) : 10  
- Nous-Hermes-Llama2 (using oobabooga/text-generation-webui) : 8.5  
- 13B-BlueMethod (using oobabooga/text-generation-webui) : 10  
- MythoBoros-13B (using oobabooga/text-generation-webui) : 10  
- WizardLM-13B-V1.2 (using oobabooga/text-generation-webui) : 10  
- 13B-Ouroboros (using oobabooga/text-generation-webui) : 10  
- AlpacaCielo-13B (using oobabooga/text-generation-webui) : 10  
- OpenChat_v3.2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-2.0 (using oobabooga/text-generation-webui) : 10  
- StableBeluga-13B (using oobabooga/text-generation-webui) : 10  
- Kimiko-13B (using oobabooga/text-generation-webui) : 3  
- Chronos-13B-v2 (using oobabooga/text-generation-webui) : 10  
- WizardLM-1.0-Uncensored-Llama2-13B (using oobabooga/text-generation-webui) : 10  
- Huginn-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-Beluga-v2-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-Hermes-13B-v2 (using oobabooga/text-generation-webui) : 10  
- vicuna-13B-v1.5 (using oobabooga/text-generation-webui) : 10  
- Stable-Platypus2-13B (using oobabooga/text-generation-webui) : 8.5  
- Platypus2-13B (using oobabooga/text-generation-webui) : 9  
- Camel-Platypus2-13B (using oobabooga/text-generation-webui) : 4  
- OpenOrca-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- PuddleJumper-13B (using oobabooga/text-generation-webui) : 10  
- Kimiko-v2-13B (using oobabooga/text-generation-webui) : 10  
- Mythical-Destroyer-V2-L2-13B (using oobabooga/text-generation-webui) : 9.5  
- Athena-v1 (using oobabooga/text-generation-webui) : 10  
- Luban-13B (using oobabooga/text-generation-webui) : 10  
- OpenBuddy-Llama2-13B-v11.1 (using oobabooga/text-generation-webui) : 10  
- Kuchiki-1.1-L2-7B (using oobabooga/text-generation-webui) : 10  
- Pygmalion-2-13B-SuperCOT2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-m2.0 (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-OpenOrca (using oobabooga/text-generation-webui) : 10  
- Nous-Capybara-7B (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-Instruct-v0.1 (using oobabooga/text-generation-webui) : 10  

### Question 11: Rewrite the following sentence in passive voice: "The dog chased the cat."  
Task Domain: Text Rewriting  
Expected Good Response: "The cat was chased by the dog."  
Explanation: This task tests the model's ability to manipulate and rewrite text according to specific grammatical requirements.  
- wizardLM-7B.q4_2 (in GPT4All) : 10  
- gpt4all-j-v1.3-groovy (in GPT4All) : 3  
- vicuna-13b-1.1-q4_2 (in GPT4All) : 10  
- gpt4-x-alpaca-13b-ggml-q4_0 (using llama.cpp) : 10  
- koala-13B-4bit-128g.GGML (using llama.cpp) : 10  
- wizard-vicuna-13B.ggml.q4_0 (using llama.cpp) : 10  
- mpt-7b-chat (in GPT4All) : 10  
- stable-vicuna-13B-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 10  
- mpt-7b-instruct : 10  
- wizard-lm-uncensored-13b-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 10  
- Manticore-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- manticore_13b_chat_pyg_GPTQ (using oobabooga/text-generation-webui) : 10  
- Project-Baize-v2-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- Airoboros-13B-GPTQ-4bit (using oobabooga/text-generation-webui) : 10  
- guanaco-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- wizardLM-13B-1.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- falcon-7b-instruct-GPTQ  (using oobabooga/text-generation-webui) : 10  
- WizardLM-Uncensored-Falcon-7B-GPTQ   (using oobabooga/text-generation-webui) : 10  
- Nous-Hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- 13B-HyperMantis_GPTQ (using oobabooga/text-generation-webui) : 10  
- Selfee-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- minotaur-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- Vicuna-13B-CoT-GPTQ (using oobabooga/text-generation-webui) : 10  
- CAMEL_13B_Combined_Data_GPTQ (using oobabooga/text-generation-webui) : 10  
- tulu-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- chronos-hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- robin-13B-v2-GPTQ (using oobabooga/text-generation-webui) : 10  
- minotaur-15B-GPTQ (using oobabooga/text-generation-webui) : 10  
- orca-mini-13b.ggmlv3.q5_K_M (using oobabooga/text-generation-webui) : 10  
- vicuna-13b-v1.3.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- vicuna-7B-v1.3-GPTQ (using oobabooga/text-generation-webui) : 10  
- UltraLM-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- airoboros-13B-gpt4-1.4-GPTQ (using oobabooga/text-generation-webui) : 10  
- llama-13b-supercot-GGML (using oobabooga/text-generation-webui) : 10  
- baichuan-vicuna-7B-GGML (using oobabooga/text-generation-webui) : 6.5  
- WizardLM-13B-V1.1 (using oobabooga/text-generation-webui) : 10  
- Llama-2-13B-chat (using oobabooga/text-generation-webui) : 10  
- Luna-AI-Llama2-Uncensored (using oobabooga/text-generation-webui) : 10  
- MythoLogic-13B (using oobabooga/text-generation-webui) : 10  
- Redmond-Puffin-13B (using oobabooga/text-generation-webui) : 10  
- llama-2-13B-Guanaco-QLoRA (using oobabooga/text-generation-webui) : 10  
- Dolphin-Llama-13B (using oobabooga/text-generation-webui) : 10  
- Nous-Hermes-Llama2 (using oobabooga/text-generation-webui) : 10  
- 13B-BlueMethod (using oobabooga/text-generation-webui) : 10  
- MythoBoros-13B (using oobabooga/text-generation-webui) : 10  
- WizardLM-13B-V1.2 (using oobabooga/text-generation-webui) : 10  
- 13B-Ouroboros (using oobabooga/text-generation-webui) : 10  
- AlpacaCielo-13B (using oobabooga/text-generation-webui) : 10  
- OpenChat_v3.2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-2.0 (using oobabooga/text-generation-webui) : 10  
- StableBeluga-13B (using oobabooga/text-generation-webui) : 10  
- Kimiko-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-13B-v2 (using oobabooga/text-generation-webui) : 10  
- WizardLM-1.0-Uncensored-Llama2-13B (using oobabooga/text-generation-webui) : 10  
- Huginn-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-Beluga-v2-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-Hermes-13B-v2 (using oobabooga/text-generation-webui) : 10  
- vicuna-13B-v1.5 (using oobabooga/text-generation-webui) : 10  
- Stable-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- Camel-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- OpenOrca-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- PuddleJumper-13B (using oobabooga/text-generation-webui) : 10  
- Kimiko-v2-13B (using oobabooga/text-generation-webui) : 9.5  
- Mythical-Destroyer-V2-L2-13B (using oobabooga/text-generation-webui) : 7  
- Athena-v1 (using oobabooga/text-generation-webui) : 10  
- Luban-13B (using oobabooga/text-generation-webui) : 10  
- OpenBuddy-Llama2-13B-v11.1 (using oobabooga/text-generation-webui) : 10  
- Kuchiki-1.1-L2-7B (using oobabooga/text-generation-webui) : 10  
- Pygmalion-2-13B-SuperCOT2 (using oobabooga/text-generation-webui) : 7.34  
- airoboros-l2-13b-gpt4-m2.0 (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-OpenOrca (using oobabooga/text-generation-webui) : 10  
- Nous-Capybara-7B (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-Instruct-v0.1 (using oobabooga/text-generation-webui) : 10  

### Question 12: Use the retrieved context to answer a question. Context does not contain the answer. (Prompt too long. See spreadsheet for original prompt)
Task Domain: Document Question Answering  
Expected Good Response: "I don't know."  
Explanation: This task tests the model's ability to understand context and answer questions based on the information provided in the context.  
- wizardLM-7B.q4_2 (in GPT4All) : 10  
- gpt4all-j-v1.3-groovy (in GPT4All) : 1  
- vicuna-13b-1.1-q4_2 (in GPT4All) : 1  
- gpt4-x-alpaca-13b-ggml-q4_0 (using llama.cpp) : 1  
- koala-13B-4bit-128g.GGML (using llama.cpp) : 10  
- wizard-vicuna-13B.ggml.q4_0 (using llama.cpp) : 10  
- mpt-7b-chat (in GPT4All) : 10  
- stable-vicuna-13B-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 10  
- mpt-7b-instruct : 1  
- wizard-lm-uncensored-13b-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 10  
- Manticore-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- manticore_13b_chat_pyg_GPTQ (using oobabooga/text-generation-webui) : 1  
- Project-Baize-v2-13B-GPTQ (using oobabooga/text-generation-webui) : 3  
- Airoboros-13B-GPTQ-4bit (using oobabooga/text-generation-webui) : 10  
- guanaco-13B-GPTQ (using oobabooga/text-generation-webui) : 1  
- wizardLM-13B-1.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- falcon-7b-instruct-GPTQ  (using oobabooga/text-generation-webui) : 1  
- WizardLM-Uncensored-Falcon-7B-GPTQ   (using oobabooga/text-generation-webui) : 1  
- Nous-Hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- 13B-HyperMantis_GPTQ (using oobabooga/text-generation-webui) : 10  
- Selfee-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- minotaur-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- Vicuna-13B-CoT-GPTQ (using oobabooga/text-generation-webui) : 10  
- CAMEL_13B_Combined_Data_GPTQ (using oobabooga/text-generation-webui) : 1  
- tulu-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- chronos-hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- robin-13B-v2-GPTQ (using oobabooga/text-generation-webui) : 10  
- minotaur-15B-GPTQ (using oobabooga/text-generation-webui) : 1  
- orca-mini-13b.ggmlv3.q5_K_M (using oobabooga/text-generation-webui) : 10  
- vicuna-13b-v1.3.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- vicuna-7B-v1.3-GPTQ (using oobabooga/text-generation-webui) : 10  
- UltraLM-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- airoboros-13B-gpt4-1.4-GPTQ (using oobabooga/text-generation-webui) : 10  
- llama-13b-supercot-GGML (using oobabooga/text-generation-webui) : 1  
- baichuan-vicuna-7B-GGML (using oobabooga/text-generation-webui) : 7  
- WizardLM-13B-V1.1 (using oobabooga/text-generation-webui) : 10  
- Llama-2-13B-chat (using oobabooga/text-generation-webui) : 10  
- Luna-AI-Llama2-Uncensored (using oobabooga/text-generation-webui) : 1  
- MythoLogic-13B (using oobabooga/text-generation-webui) : 10  
- Redmond-Puffin-13B (using oobabooga/text-generation-webui) : 1  
- llama-2-13B-Guanaco-QLoRA (using oobabooga/text-generation-webui) : 10  
- Dolphin-Llama-13B (using oobabooga/text-generation-webui) : 10  
- Nous-Hermes-Llama2 (using oobabooga/text-generation-webui) : 10  
- 13B-BlueMethod (using oobabooga/text-generation-webui) : 1  
- MythoBoros-13B (using oobabooga/text-generation-webui) : 10  
- WizardLM-13B-V1.2 (using oobabooga/text-generation-webui) : 10  
- 13B-Ouroboros (using oobabooga/text-generation-webui) : 10  
- AlpacaCielo-13B (using oobabooga/text-generation-webui) : 10  
- OpenChat_v3.2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-2.0 (using oobabooga/text-generation-webui) : 10  
- StableBeluga-13B (using oobabooga/text-generation-webui) : 10  
- Kimiko-13B (using oobabooga/text-generation-webui) : 1  
- Chronos-13B-v2 (using oobabooga/text-generation-webui) : 10  
- WizardLM-1.0-Uncensored-Llama2-13B (using oobabooga/text-generation-webui) : 10  
- Huginn-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-Beluga-v2-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-Hermes-13B-v2 (using oobabooga/text-generation-webui) : 10  
- vicuna-13B-v1.5 (using oobabooga/text-generation-webui) : 10  
- Stable-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- Platypus2-13B (using oobabooga/text-generation-webui) : 1  
- Camel-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- OpenOrca-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- PuddleJumper-13B (using oobabooga/text-generation-webui) : 10  
- Kimiko-v2-13B (using oobabooga/text-generation-webui) : 1  
- Mythical-Destroyer-V2-L2-13B (using oobabooga/text-generation-webui) : 2  
- Athena-v1 (using oobabooga/text-generation-webui) : 10  
- Luban-13B (using oobabooga/text-generation-webui) : 10  
- OpenBuddy-Llama2-13B-v11.1 (using oobabooga/text-generation-webui) : 10  
- Kuchiki-1.1-L2-7B (using oobabooga/text-generation-webui) : 1  
- Pygmalion-2-13B-SuperCOT2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-m2.0 (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-OpenOrca (using oobabooga/text-generation-webui) : 10  
- Nous-Capybara-7B (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-Instruct-v0.1 (using oobabooga/text-generation-webui) : 10  

### Question 13: Use the retrieved context to answer a question. Context contains the answer. (Prompt too long. See spreadsheet for original prompt)
Task Domain: Document Question Answering  
Expected Good Response: "["Semantic text search", "Generative question-answering", "Hybrid search", "Image similarity search", "Product recommendations"]"  
Explanation: This task tests the model's ability to understand context and answer questions based on the information provided in the context.  
- wizardLM-7B.q4_2 (in GPT4All) : 10  
- gpt4all-j-v1.3-groovy (in GPT4All) : 1  
- vicuna-13b-1.1-q4_2 (in GPT4All) : 1  
- gpt4-x-alpaca-13b-ggml-q4_0 (using llama.cpp) : 4  
- koala-13B-4bit-128g.GGML (using llama.cpp) : 5.5  
- wizard-vicuna-13B.ggml.q4_0 (using llama.cpp) : 10  
- mpt-7b-chat (in GPT4All) : 4  
- stable-vicuna-13B-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 1  
- mpt-7b-instruct : 10  
- wizard-lm-uncensored-13b-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 5  
- Manticore-13B-GPTQ (using oobabooga/text-generation-webui) : 2  
- manticore_13b_chat_pyg_GPTQ (using oobabooga/text-generation-webui) : 5  
- Project-Baize-v2-13B-GPTQ (using oobabooga/text-generation-webui) : 1  
- Airoboros-13B-GPTQ-4bit (using oobabooga/text-generation-webui) : 7  
- guanaco-13B-GPTQ (using oobabooga/text-generation-webui) : 8  
- wizardLM-13B-1.0-GPTQ (using oobabooga/text-generation-webui) : 8.5  
- falcon-7b-instruct-GPTQ  (using oobabooga/text-generation-webui) : 10  
- WizardLM-Uncensored-Falcon-7B-GPTQ   (using oobabooga/text-generation-webui) : 7.5  
- Nous-Hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 5  
- 13B-HyperMantis_GPTQ (using oobabooga/text-generation-webui) : 10  
- Selfee-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- minotaur-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- Vicuna-13B-CoT-GPTQ (using oobabooga/text-generation-webui) : 1  
- CAMEL_13B_Combined_Data_GPTQ (using oobabooga/text-generation-webui) : 8.5  
- tulu-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- chronos-hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 5.7  
- robin-13B-v2-GPTQ (using oobabooga/text-generation-webui) : 6.5  
- minotaur-15B-GPTQ (using oobabooga/text-generation-webui) : 1  
- orca-mini-13b.ggmlv3.q5_K_M (using oobabooga/text-generation-webui) : 7.4  
- vicuna-13b-v1.3.0-GPTQ (using oobabooga/text-generation-webui) : 5  
- vicuna-7B-v1.3-GPTQ (using oobabooga/text-generation-webui) : 1  
- UltraLM-13B-GPTQ (using oobabooga/text-generation-webui) : 7.7  
- airoboros-13B-gpt4-1.4-GPTQ (using oobabooga/text-generation-webui) : 10  
- llama-13b-supercot-GGML (using oobabooga/text-generation-webui) : 5.6875  
- baichuan-vicuna-7B-GGML (using oobabooga/text-generation-webui) : 1  
- WizardLM-13B-V1.1 (using oobabooga/text-generation-webui) : 1  
- Llama-2-13B-chat (using oobabooga/text-generation-webui) : 10  
- Luna-AI-Llama2-Uncensored (using oobabooga/text-generation-webui) : 7.5  
- MythoLogic-13B (using oobabooga/text-generation-webui) : 7.4  
- Redmond-Puffin-13B (using oobabooga/text-generation-webui) : 10  
- llama-2-13B-Guanaco-QLoRA (using oobabooga/text-generation-webui) : 10  
- Dolphin-Llama-13B (using oobabooga/text-generation-webui) : 10  
- Nous-Hermes-Llama2 (using oobabooga/text-generation-webui) : 5.6875  
- 13B-BlueMethod (using oobabooga/text-generation-webui) : 7.4  
- MythoBoros-13B (using oobabooga/text-generation-webui) : 7.34  
- WizardLM-13B-V1.2 (using oobabooga/text-generation-webui) : 1  
- 13B-Ouroboros (using oobabooga/text-generation-webui) : 10  
- AlpacaCielo-13B (using oobabooga/text-generation-webui) : 10  
- OpenChat_v3.2 (using oobabooga/text-generation-webui) : 1  
- airoboros-l2-13b-gpt4-2.0 (using oobabooga/text-generation-webui) : 10  
- StableBeluga-13B (using oobabooga/text-generation-webui) : 10  
- Kimiko-13B (using oobabooga/text-generation-webui) : 3  
- Chronos-13B-v2 (using oobabooga/text-generation-webui) : 7.5  
- WizardLM-1.0-Uncensored-Llama2-13B (using oobabooga/text-generation-webui) : 1  
- Huginn-13B (using oobabooga/text-generation-webui) : 8.5  
- Chronos-Beluga-v2-13B (using oobabooga/text-generation-webui) : 7.5  
- Chronos-Hermes-13B-v2 (using oobabooga/text-generation-webui) : 5  
- vicuna-13B-v1.5 (using oobabooga/text-generation-webui) : 10  
- Stable-Platypus2-13B (using oobabooga/text-generation-webui) : 1  
- Platypus2-13B (using oobabooga/text-generation-webui) : 8  
- Camel-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- OpenOrca-Platypus2-13B (using oobabooga/text-generation-webui) : 7.4  
- PuddleJumper-13B (using oobabooga/text-generation-webui) : 4  
- Kimiko-v2-13B (using oobabooga/text-generation-webui) : 2  
- Mythical-Destroyer-V2-L2-13B (using oobabooga/text-generation-webui) : 2  
- Athena-v1 (using oobabooga/text-generation-webui) : 3  
- Luban-13B (using oobabooga/text-generation-webui) : 1  
- OpenBuddy-Llama2-13B-v11.1 (using oobabooga/text-generation-webui) : 1  
- Kuchiki-1.1-L2-7B (using oobabooga/text-generation-webui) : 10  
- Pygmalion-2-13B-SuperCOT2 (using oobabooga/text-generation-webui) : 3  
- airoboros-l2-13b-gpt4-m2.0 (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-OpenOrca (using oobabooga/text-generation-webui) : 10  
- Nous-Capybara-7B (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-Instruct-v0.1 (using oobabooga/text-generation-webui) : 10  

### Question 14: What is the square root of banana?
Task Domain: Mathematical Reasoning  
Expected Good Response: "The question is nonsensical, as square roots can only be calculated for numbers, not for objects or words like 'banana'."  
Explanation: This task tests the model's ability to recognize questions that are illogical or nonsensical and respond accordingly.  
- wizardLM-7B.q4_2 (in GPT4All) : 10  
- gpt4all-j-v1.3-groovy (in GPT4All) : 1  
- vicuna-13b-1.1-q4_2 (in GPT4All) : 10  
- gpt4-x-alpaca-13b-ggml-q4_0 (using llama.cpp) : 1  
- koala-13B-4bit-128g.GGML (using llama.cpp) : 1  
- wizard-vicuna-13B.ggml.q4_0 (using llama.cpp) : 10  
- mpt-7b-chat (in GPT4All) : 10  
- stable-vicuna-13B-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 10  
- mpt-7b-instruct : 1  
- wizard-lm-uncensored-13b-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 1  
- Manticore-13B-GPTQ (using oobabooga/text-generation-webui) : 1  
- manticore_13b_chat_pyg_GPTQ (using oobabooga/text-generation-webui) : 10  
- Project-Baize-v2-13B-GPTQ (using oobabooga/text-generation-webui) : 4  
- Airoboros-13B-GPTQ-4bit (using oobabooga/text-generation-webui) : 10  
- guanaco-13B-GPTQ (using oobabooga/text-generation-webui) : 1  
- wizardLM-13B-1.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- falcon-7b-instruct-GPTQ  (using oobabooga/text-generation-webui) : 1  
- WizardLM-Uncensored-Falcon-7B-GPTQ   (using oobabooga/text-generation-webui) : 7.5  
- Nous-Hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- 13B-HyperMantis_GPTQ (using oobabooga/text-generation-webui) : 10  
- Selfee-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- minotaur-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- Vicuna-13B-CoT-GPTQ (using oobabooga/text-generation-webui) : 10  
- CAMEL_13B_Combined_Data_GPTQ (using oobabooga/text-generation-webui) : 10  
- tulu-13B-GPTQ (using oobabooga/text-generation-webui) : 1  
- chronos-hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- robin-13B-v2-GPTQ (using oobabooga/text-generation-webui) : 4  
- minotaur-15B-GPTQ (using oobabooga/text-generation-webui) : 1  
- orca-mini-13b.ggmlv3.q5_K_M (using oobabooga/text-generation-webui) : 1  
- vicuna-13b-v1.3.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- vicuna-7B-v1.3-GPTQ (using oobabooga/text-generation-webui) : 2  
- UltraLM-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- airoboros-13B-gpt4-1.4-GPTQ (using oobabooga/text-generation-webui) : 10  
- llama-13b-supercot-GGML (using oobabooga/text-generation-webui) : 4.5  
- baichuan-vicuna-7B-GGML (using oobabooga/text-generation-webui) : 4  
- WizardLM-13B-V1.1 (using oobabooga/text-generation-webui) : 10  
- Llama-2-13B-chat (using oobabooga/text-generation-webui) : 10  
- Luna-AI-Llama2-Uncensored (using oobabooga/text-generation-webui) : 10  
- MythoLogic-13B (using oobabooga/text-generation-webui) : 10  
- Redmond-Puffin-13B (using oobabooga/text-generation-webui) : 1  
- llama-2-13B-Guanaco-QLoRA (using oobabooga/text-generation-webui) : 10  
- Dolphin-Llama-13B (using oobabooga/text-generation-webui) : 10  
- Nous-Hermes-Llama2 (using oobabooga/text-generation-webui) : 8.5  
- 13B-BlueMethod (using oobabooga/text-generation-webui) : 1  
- MythoBoros-13B (using oobabooga/text-generation-webui) : 10  
- WizardLM-13B-V1.2 (using oobabooga/text-generation-webui) : 10  
- 13B-Ouroboros (using oobabooga/text-generation-webui) : 7.25  
- AlpacaCielo-13B (using oobabooga/text-generation-webui) : 8.5  
- OpenChat_v3.2 (using oobabooga/text-generation-webui) : 4  
- airoboros-l2-13b-gpt4-2.0 (using oobabooga/text-generation-webui) : 10  
- StableBeluga-13B (using oobabooga/text-generation-webui) : 7.5  
- Kimiko-13B (using oobabooga/text-generation-webui) : 1  
- Chronos-13B-v2 (using oobabooga/text-generation-webui) : 9.5  
- WizardLM-1.0-Uncensored-Llama2-13B (using oobabooga/text-generation-webui) : 7.5  
- Huginn-13B (using oobabooga/text-generation-webui) : 6.5  
- Chronos-Beluga-v2-13B (using oobabooga/text-generation-webui) : 4.5  
- Chronos-Hermes-13B-v2 (using oobabooga/text-generation-webui) : 5.5  
- vicuna-13B-v1.5 (using oobabooga/text-generation-webui) : 10  
- Stable-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- Platypus2-13B (using oobabooga/text-generation-webui) : 1  
- Camel-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- OpenOrca-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- PuddleJumper-13B (using oobabooga/text-generation-webui) : 7.5  
- Kimiko-v2-13B (using oobabooga/text-generation-webui) : 1  
- Mythical-Destroyer-V2-L2-13B (using oobabooga/text-generation-webui) : 7.5  
- Athena-v1 (using oobabooga/text-generation-webui) : 10  
- Luban-13B (using oobabooga/text-generation-webui) : 9.5  
- OpenBuddy-Llama2-13B-v11.1 (using oobabooga/text-generation-webui) : 6.5  
- Kuchiki-1.1-L2-7B (using oobabooga/text-generation-webui) : 10  
- Pygmalion-2-13B-SuperCOT2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-m2.0 (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-OpenOrca (using oobabooga/text-generation-webui) : 10  
- Nous-Capybara-7B (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-Instruct-v0.1 (using oobabooga/text-generation-webui) : 4  

### Question 15: Extract the sender's name and address from the following text: "Dear Troy, Thanks for sharing your thoughts on document qa with Claude LLM and your comments on Tim's thoughts. My address is 5000 Forbes Ave, Pittsburgh, PA 15213. Best, Alex." Respond in JSON with one field for name and the other field for address.
Task Domain: Information Extraction  
Expected Good Response: {"name": "Alex","address": "5000 Forbes Ave, Pittsburgh, PA 15213"}  
Explanation: This task tests the model's ability to extract specific information (sender's name and address) from a given text and present the extracted information in a JSON format.  
- wizardLM-7B.q4_2 (in GPT4All) : 10  
- gpt4all-j-v1.3-groovy (in GPT4All) : 10  
- vicuna-13b-1.1-q4_2 (in GPT4All) : 10  
- gpt4-x-alpaca-13b-ggml-q4_0 (using llama.cpp) : 10  
- koala-13B-4bit-128g.GGML (using llama.cpp) : 10  
- wizard-vicuna-13B.ggml.q4_0 (using llama.cpp) : 10  
- mpt-7b-chat (in GPT4All) : 10  
- stable-vicuna-13B-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 10  
- mpt-7b-instruct : 6  
- wizard-lm-uncensored-13b-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 10  
- Manticore-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- manticore_13b_chat_pyg_GPTQ (using oobabooga/text-generation-webui) : 8  
- Project-Baize-v2-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- Airoboros-13B-GPTQ-4bit (using oobabooga/text-generation-webui) : 1  
- guanaco-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- wizardLM-13B-1.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- falcon-7b-instruct-GPTQ  (using oobabooga/text-generation-webui) : 10  
- WizardLM-Uncensored-Falcon-7B-GPTQ   (using oobabooga/text-generation-webui) : 10  
- Nous-Hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- 13B-HyperMantis_GPTQ (using oobabooga/text-generation-webui) : 8  
- Selfee-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- minotaur-13B-GPTQ (using oobabooga/text-generation-webui) : 7.5  
- Vicuna-13B-CoT-GPTQ (using oobabooga/text-generation-webui) : 10  
- CAMEL_13B_Combined_Data_GPTQ (using oobabooga/text-generation-webui) : 10  
- tulu-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- chronos-hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- robin-13B-v2-GPTQ (using oobabooga/text-generation-webui) : 10  
- minotaur-15B-GPTQ (using oobabooga/text-generation-webui) : 6  
- orca-mini-13b.ggmlv3.q5_K_M (using oobabooga/text-generation-webui) : 10  
- vicuna-13b-v1.3.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- vicuna-7B-v1.3-GPTQ (using oobabooga/text-generation-webui) : 10  
- UltraLM-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- airoboros-13B-gpt4-1.4-GPTQ (using oobabooga/text-generation-webui) : 10  
- llama-13b-supercot-GGML (using oobabooga/text-generation-webui) : 10  
- baichuan-vicuna-7B-GGML (using oobabooga/text-generation-webui) : 7  
- WizardLM-13B-V1.1 (using oobabooga/text-generation-webui) : 4  
- Llama-2-13B-chat (using oobabooga/text-generation-webui) : 10  
- Luna-AI-Llama2-Uncensored (using oobabooga/text-generation-webui) : 10  
- MythoLogic-13B (using oobabooga/text-generation-webui) : 10  
- Redmond-Puffin-13B (using oobabooga/text-generation-webui) : 10  
- llama-2-13B-Guanaco-QLoRA (using oobabooga/text-generation-webui) : 4  
- Dolphin-Llama-13B (using oobabooga/text-generation-webui) : 8.5  
- Nous-Hermes-Llama2 (using oobabooga/text-generation-webui) : 10  
- 13B-BlueMethod (using oobabooga/text-generation-webui) : 10  
- MythoBoros-13B (using oobabooga/text-generation-webui) : 10  
- WizardLM-13B-V1.2 (using oobabooga/text-generation-webui) : 10  
- 13B-Ouroboros (using oobabooga/text-generation-webui) : 10  
- AlpacaCielo-13B (using oobabooga/text-generation-webui) : 4  
- OpenChat_v3.2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-2.0 (using oobabooga/text-generation-webui) : 10  
- StableBeluga-13B (using oobabooga/text-generation-webui) : 4  
- Kimiko-13B (using oobabooga/text-generation-webui) : 10  
- Chronos-13B-v2 (using oobabooga/text-generation-webui) : 10  
- WizardLM-1.0-Uncensored-Llama2-13B (using oobabooga/text-generation-webui) : 10  
- Huginn-13B (using oobabooga/text-generation-webui) : 3  
- Chronos-Beluga-v2-13B (using oobabooga/text-generation-webui) : 6  
- Chronos-Hermes-13B-v2 (using oobabooga/text-generation-webui) : 10  
- vicuna-13B-v1.5 (using oobabooga/text-generation-webui) : 4  
- Stable-Platypus2-13B (using oobabooga/text-generation-webui) : 10  
- Platypus2-13B (using oobabooga/text-generation-webui) : 4  
- Camel-Platypus2-13B (using oobabooga/text-generation-webui) : 4  
- OpenOrca-Platypus2-13B (using oobabooga/text-generation-webui) : 4  
- PuddleJumper-13B (using oobabooga/text-generation-webui) : 10  
- Kimiko-v2-13B (using oobabooga/text-generation-webui) : 10  
- Mythical-Destroyer-V2-L2-13B (using oobabooga/text-generation-webui) : 8.5  
- Athena-v1 (using oobabooga/text-generation-webui) : 10  
- Luban-13B (using oobabooga/text-generation-webui) : 10  
- OpenBuddy-Llama2-13B-v11.1 (using oobabooga/text-generation-webui) : 10  
- Kuchiki-1.1-L2-7B (using oobabooga/text-generation-webui) : 10  
- Pygmalion-2-13B-SuperCOT2 (using oobabooga/text-generation-webui) : 10  
- airoboros-l2-13b-gpt4-m2.0 (using oobabooga/text-generation-webui) : 4  
- Mistral-7B-OpenOrca (using oobabooga/text-generation-webui) : 10  
- Nous-Capybara-7B (using oobabooga/text-generation-webui) : 10  
- Mistral-7B-Instruct-v0.1 (using oobabooga/text-generation-webui) : 10  

### Question 16: Given the following list of words. Categorize the words into 5 categories by similarity. Give each category a name. Respond in a python dictionary with key as the category name and value as a list of words in that category. List of words: ['Quagmire', 'Luminous', 'Melancholy', 'Perplexed', 'Jubilant', 'Enigmatic', 'Ambiguous', 'Ravenous', 'Obsolete', 'Tenacious', 'Euphoric', 'Wistful', 'Clandestine', 'Insidious', 'Inquisitive', 'Resilient', 'Surreptitious', 'Serendipity', 'Idiosyncratic', 'Juxtaposition']
Task Domain: Categorization  
Expected Good Response:  
{
  "Emotions": ['Melancholy', 'Jubilant', 'Euphoric', 'Wistful'],
  "Qualities": ['Luminous', 'Tenacious', 'Resilient'],
  "Mysterious": ['Quagmire', 'Enigmatic', 'Ambiguous', 'Clandestine', 'Surreptitious'],
  "Inquisitive": ['Perplexed', 'Inquisitive'],
  "Uncommon": ['Ravenous', 'Obsolete', 'Insidious', 'Serendipity', 'Idiosyncratic', 'Juxtaposition']
}  
Explanation: This task tests the model's ability to categorize a list of words into groups based on their similarity and provide appropriate category names. The response is in a Python dictionary format as specified in the question.  
- wizardLM-7B.q4_2 (in GPT4All) : 1  
- gpt4all-j-v1.3-groovy (in GPT4All) : 1  
- vicuna-13b-1.1-q4_2 (in GPT4All) : 4  
- gpt4-x-alpaca-13b-ggml-q4_0 (using llama.cpp) : 3  
- koala-13B-4bit-128g.GGML (using llama.cpp) : 1  
- wizard-vicuna-13B.ggml.q4_0 (using llama.cpp) : 4  
- mpt-7b-chat (in GPT4All) : 3  
- stable-vicuna-13B-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 4  
- mpt-7b-instruct : 7  
- wizard-lm-uncensored-13b-GPTQ-4bit-128g (using oobabooga/text-generation-webui) : 5  
- Manticore-13B-GPTQ (using oobabooga/text-generation-webui) : 3  
- manticore_13b_chat_pyg_GPTQ (using oobabooga/text-generation-webui) : 1  
- Project-Baize-v2-13B-GPTQ (using oobabooga/text-generation-webui) : 4  
- Airoboros-13B-GPTQ-4bit (using oobabooga/text-generation-webui) : 6  
- guanaco-13B-GPTQ (using oobabooga/text-generation-webui) : 5  
- wizardLM-13B-1.0-GPTQ (using oobabooga/text-generation-webui) : 7  
- falcon-7b-instruct-GPTQ  (using oobabooga/text-generation-webui) : 1  
- WizardLM-Uncensored-Falcon-7B-GPTQ   (using oobabooga/text-generation-webui) : 3  
- Nous-Hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 7  
- 13B-HyperMantis_GPTQ (using oobabooga/text-generation-webui) : 3  
- Selfee-13B-GPTQ (using oobabooga/text-generation-webui) : 4  
- minotaur-13B-GPTQ (using oobabooga/text-generation-webui) : 2  
- Vicuna-13B-CoT-GPTQ (using oobabooga/text-generation-webui) : 2  
- CAMEL_13B_Combined_Data_GPTQ (using oobabooga/text-generation-webui) : 4  
- tulu-13B-GPTQ (using oobabooga/text-generation-webui) : 4  
- chronos-hermes-13B-GPTQ (using oobabooga/text-generation-webui) : 3  
- robin-13B-v2-GPTQ (using oobabooga/text-generation-webui) : 4  
- minotaur-15B-GPTQ (using oobabooga/text-generation-webui) : 1  
- orca-mini-13b.ggmlv3.q5_K_M (using oobabooga/text-generation-webui) : 4  
- vicuna-13b-v1.3.0-GPTQ (using oobabooga/text-generation-webui) : 4  
- vicuna-7B-v1.3-GPTQ (using oobabooga/text-generation-webui) : 4  
- UltraLM-13B-GPTQ (using oobabooga/text-generation-webui) : 5  
- airoboros-13B-gpt4-1.4-GPTQ (using oobabooga/text-generation-webui) : 5  
- llama-13b-supercot-GGML (using oobabooga/text-generation-webui) : 4.5  
- baichuan-vicuna-7B-GGML (using oobabooga/text-generation-webui) : 3  
- WizardLM-13B-V1.1 (using oobabooga/text-generation-webui) : 5  
- Llama-2-13B-chat (using oobabooga/text-generation-webui) : 7.4  
- Luna-AI-Llama2-Uncensored (using oobabooga/text-generation-webui) : 4  
- MythoLogic-13B (using oobabooga/text-generation-webui) : 4  
- Redmond-Puffin-13B (using oobabooga/text-generation-webui) : 4  
- llama-2-13B-Guanaco-QLoRA (using oobabooga/text-generation-webui) : 3  
- Dolphin-Llama-13B (using oobabooga/text-generation-webui) : 4  
- Nous-Hermes-Llama2 (using oobabooga/text-generation-webui) : 4  
- 13B-BlueMethod (using oobabooga/text-generation-webui) : 4  
- MythoBoros-13B (using oobabooga/text-generation-webui) : 4  
- WizardLM-13B-V1.2 (using oobabooga/text-generation-webui) : 7.5  
- 13B-Ouroboros (using oobabooga/text-generation-webui) : 4  
- AlpacaCielo-13B (using oobabooga/text-generation-webui) : 4  
- OpenChat_v3.2 (using oobabooga/text-generation-webui) : 4.5  
- airoboros-l2-13b-gpt4-2.0 (using oobabooga/text-generation-webui) : 4  
- StableBeluga-13B (using oobabooga/text-generation-webui) : 5  
- Kimiko-13B (using oobabooga/text-generation-webui) : 3  
- Chronos-13B-v2 (using oobabooga/text-generation-webui) : 3  
- WizardLM-1.0-Uncensored-Llama2-13B (using oobabooga/text-generation-webui) : 7.4  
- Huginn-13B (using oobabooga/text-generation-webui) : 4  
- Chronos-Beluga-v2-13B (using oobabooga/text-generation-webui) : 2  
- Chronos-Hermes-13B-v2 (using oobabooga/text-generation-webui) : 3  
- vicuna-13B-v1.5 (using oobabooga/text-generation-webui) : 8.5  
- Stable-Platypus2-13B (using oobabooga/text-generation-webui) : 4  
- Platypus2-13B (using oobabooga/text-generation-webui) : 2  
- Camel-Platypus2-13B (using oobabooga/text-generation-webui) : 8.5  
- OpenOrca-Platypus2-13B (using oobabooga/text-generation-webui) : 8.5  
- PuddleJumper-13B (using oobabooga/text-generation-webui) : 8.5  
- Kimiko-v2-13B (using oobabooga/text-generation-webui) : 2  
- Mythical-Destroyer-V2-L2-13B (using oobabooga/text-generation-webui) : 7.5  
- Athena-v1 (using oobabooga/text-generation-webui) : 5  
- Luban-13B (using oobabooga/text-generation-webui) : 8.5  
- OpenBuddy-Llama2-13B-v11.1 (using oobabooga/text-generation-webui) : 8.5  
- Kuchiki-1.1-L2-7B (using oobabooga/text-generation-webui) : 1  
- Pygmalion-2-13B-SuperCOT2 (using oobabooga/text-generation-webui) : 7.4  
- airoboros-l2-13b-gpt4-m2.0 (using oobabooga/text-generation-webui) : 3  
- Mistral-7B-OpenOrca (using oobabooga/text-generation-webui) : 10  
- Nous-Capybara-7B (using oobabooga/text-generation-webui) : 4  
- Mistral-7B-Instruct-v0.1 (using oobabooga/text-generation-webui) : 7.4  

## Coding questions and answers

I decided to test the coding capability of some coding-specific models and top general purpose models. Original responses can be found at: https://docs.google.com/spreadsheets/d/1ogDXUiaBx3t7EpMo44aaA6U6kLXX0x2tGRgLg8CISGs/edit?usp=sharing

### Question 1: Implement a Python function that takes in a list of integers and an integer target, and returns a list of pairs whose sum is equal to the target.
Task Domain: Coding  
Expected Good Response:  
```
def find_pairs_with_sum(arr, target):
    if not arr:
        return []
    pairs = []
    seen = set()
    for num in arr:
        complement = target - num
        if complement in seen:
            pairs.append((num, complement))
        seen.add(num)
    return pairs
```
Explanation of the task: This problem tests the candidate's proficiency in Python and their ability to apply efficient algorithms for a common task in software engineering.  
- wizard-vicuna-13B-GPTQ (using oobabooga/text-generation-webui) : 10  
- WizardCoder-15B-1.0-GPTQ (using oobabooga/text-generation-webui) : 5.5  
- Redmond-Hermes-Coder-GPTQ (using oobabooga/text-generation-webui) : 10  
- starchat-beta-GPTQ (using oobabooga/text-generation-webui) : 7.5  
- wizardLM-13B-1.0-GPTQ (using oobabooga/text-generation-webui) : 10  
- WizardCoder-Guanaco-15B-V1.1 (using oobabooga/text-generation-webui) : 7.5  
- CodeUp-Llama-2-13B-Chat-HF (using oobabooga/text-generation-webui) : 3  
- CodeUp-Alpha-13B-HF (using oobabooga/text-generation-webui) : 6.5  
- WizardCoder-Python-13B-V1.0 (using oobabooga/text-generation-webui) : 9.5  
- CodeLlama-13B-oasst-sft-v10 (using oobabooga/text-generation-webui) : 10  
- CodeLlama-13B-Instruct (using oobabooga/text-generation-webui) : 10  

### Question 2: Implement a computer vision model in PyTorch that can classify images of handwritten digits using the MNIST dataset.
Task Domain: Computer Vision  
Expected Good Response:  
```
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

model = torch.nn.Sequential(
    torch.nn.Linear(28*28, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 10)
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(10):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.view(inputs.size(0), -1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```
Explanation of the task: This problem tests the candidate's understanding of computer vision, deep learning, and the PyTorch framework, reflecting their ability to create and train models for real-world applications.  
- wizard-vicuna-13B-GPTQ (using oobabooga/text-generation-webui) : 6.5  
- WizardCoder-15B-1.0-GPTQ (using oobabooga/text-generation-webui) : 7.5  
- Redmond-Hermes-Coder-GPTQ (using oobabooga/text-generation-webui) : 9.5  
- starchat-beta-GPTQ (using oobabooga/text-generation-webui) : 9.5  
- wizardLM-13B-1.0-GPTQ (using oobabooga/text-generation-webui) : 4  
- WizardCoder-Guanaco-15B-V1.1 (using oobabooga/text-generation-webui) : 9.5  
- CodeUp-Llama-2-13B-Chat-HF (using oobabooga/text-generation-webui) : 9.5  
- CodeUp-Llama-2-13B-Chat-HF (using oobabooga/text-generation-webui) : 9.5  
- WizardCoder-Python-13B-V1.0 (using oobabooga/text-generation-webui) : 9.5  
- CodeLlama-13B-oasst-sft-v10 (using oobabooga/text-generation-webui) : 9.5  
- CodeLlama-13B-Instruct (using oobabooga/text-generation-webui) : 8.5  

### Question 3: Develop a sentiment analysis model using natural language processing (NLP) techniques in Python with the help of the NLTK library.
Task Domain: Natural Language Processing  
Expected Good Response:  
```
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)
    return sentiment_score
```
Explanation of the task: This problem tests the candidate's proficiency in NLP and their ability to apply it to sentiment analysis, a popular subfield with a wide range of applications in social media analysis, chatbots, and more.  
- wizard-vicuna-13B-GPTQ (using oobabooga/text-generation-webui) : 7.5  
- WizardCoder-15B-1.0-GPTQ (using oobabooga/text-generation-webui) : 9.5  
- Redmond-Hermes-Coder-GPTQ (using oobabooga/text-generation-webui) : 7.5  
- starchat-beta-GPTQ (using oobabooga/text-generation-webui) : 7.5  
- wizardLM-13B-1.0-GPTQ (using oobabooga/text-generation-webui) : 4  
- WizardCoder-Guanaco-15B-V1.1 (using oobabooga/text-generation-webui) : 5  
- CodeUp-Llama-2-13B-Chat-HF (using oobabooga/text-generation-webui) : 5  
- CodeUp-Alpha-13B-HF (using oobabooga/text-generation-webui) : 8.5  
- WizardCoder-Python-13B-V1.0 (using oobabooga/text-generation-webui) : 9.5  
- CodeLlama-13B-oasst-sft-v10 (using oobabooga/text-generation-webui) : 9.5  
- CodeLlama-13B-Instruct (using oobabooga/text-generation-webui) : 4  

### Question 4: Implement a SwiftUI view that displays a list of items and allows the user to filter the list by a search query.
Task Domain: iOS Development
Expected Good Response:  
```
import SwiftUI

struct FilteredListView: View {
    @State private var searchText = ""

    let items = ["Apple", "Banana", "Cherry", "Date", "Fig", "Grape"]

    var filteredItems: [String] {
        items.filter { searchText.isEmpty || $0.lowercased().contains(searchText.lowercased()) }
    }

    var body: some View {
        NavigationView {
            VStack {
                TextField("Search", text: $searchText)
                    .padding(.horizontal)
                List(filteredItems, id: \.self) { item in
                    Text(item)
                }
            }
            .navigationBarTitle("Fruits")
        }
    }
}
```
Explanation of the task: This problem tests the candidate's proficiency in SwiftUI, a modern framework for building user interfaces on Apple platforms, and their ability to create a responsive and dynamic UI.  
- wizard-vicuna-13B-GPTQ (using oobabooga/text-generation-webui) : 4  
- WizardCoder-15B-1.0-GPTQ (using oobabooga/text-generation-webui) : 3  
- Redmond-Hermes-Coder-GPTQ (using oobabooga/text-generation-webui) : 5.5  
- starchat-beta-GPTQ (using oobabooga/text-generation-webui) : 5  
- wizardLM-13B-1.0-GPTQ (using oobabooga/text-generation-webui) : 3  
- WizardCoder-Guanaco-15B-V1.1 (using oobabooga/text-generation-webui) : 4  
- CodeUp-Llama-2-13B-Chat-HF (using oobabooga/text-generation-webui) : 4  
- CodeUp-Alpha-13B-HF (using oobabooga/text-generation-webui) : 6.5  
- WizardCoder-Python-13B-V1.0 (using oobabooga/text-generation-webui) : 9.5  
- CodeLlama-13B-oasst-sft-v10 (using oobabooga/text-generation-webui) : 10  
- CodeLlama-13B-Instruct (using oobabooga/text-generation-webui) : 4  

### Question 5: Design a RESTful API in Flask (Python) that allows users to create, read, update, and delete (CRUD) notes.
Task Domain: Web Development  
Expected Good Response:  
```
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///notes.db'
db = SQLAlchemy(app)
ma = Marshmallow(app)

class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)

class NoteSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Note

note_schema = NoteSchema()
notes_schema = NoteSchema(many=True)

@app.route('/notes', methods=['POST'])
def create_note():
    content = request.json['content']
    new_note = Note(content=content)
    db.session.add(new_note)
    db.session.commit()
    return note_schema.dump(new_note)

@app.route('/notes', methods=['GET'])
def get_notes():
    all_notes = Note.query.all()
    return notes_schema.dump(all_notes)

@app.route('/notes/<int:id>', methods=['GET'])
def get_note(id):
    note = Note.query.get_or_404(id)
    return note_schema.dump(note)

@app.route('/notes/<int:id>', methods=['PUT'])
def update_note(id):
    note = Note.query.get_or_404(id)
    content = request.json['content']
    note.content = content
    db.session.commit()
    return note_schema.dump(note)

@app.route('/notes/<int:id>', methods=['DELETE'])
def delete_note(id):
    note = Note.query.get_or_404(id)
    db.session.delete(note)
    db.session.commit()
    return note_schema.dump(note)

if __name__ == '__main__':
    app.run()
```
Explanation of the task: This problem tests the candidate's ability to design and implement a RESTful API using Flask, a popular web framework in Python, and assesses their understanding of CRUD operations and database management.  
- wizard-vicuna-13B-GPTQ (using oobabooga/text-generation-webui) : 8.5  
- WizardCoder-15B-1.0-GPTQ (using oobabooga/text-generation-webui) : 4  
- Redmond-Hermes-Coder-GPTQ (using oobabooga/text-generation-webui) : 9.5  
- starchat-beta-GPTQ (using oobabooga/text-generation-webui) : 8.5  
- wizardLM-13B-1.0-GPTQ (using oobabooga/text-generation-webui) : 8.5  
- WizardCoder-Guanaco-15B-V1.1 (using oobabooga/text-generation-webui) : 9.5  
- CodeUp-Llama-2-13B-Chat-HF (using oobabooga/text-generation-webui) : 8.5  
- CodeUp-Alpha-13B-HF (using oobabooga/text-generation-webui) : 8.5  
- WizardCoder-Python-13B-V1.0 (using oobabooga/text-generation-webui) : 9.5  
- CodeLlama-13B-oasst-sft-v10 (using oobabooga/text-generation-webui) : 10  
- CodeLlama-13B-Instruct (using oobabooga/text-generation-webui) : 8.5  

</details>