<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vllm website to help you get started with vllm. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has evolved into a community-driven project with contributions from both academia and industry.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantizations: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [AutoRound](https://arxiv.org/abs/2309.05516), INT4, INT8, and FP8
- Optimized CUDA kernels, including integration with FlashAttention and FlashInfer
- Speculative decoding
- Chunked prefill

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data and expert parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support for NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, Arm CPUs, and TPU. Additionally, support for diverse hardware plugins such as Intel Gaudi, IBM Spyre and Huawei Ascend.
- Prefix caching support
- Multi-LoRA support

vLLM seamlessly supports most popular open-source models on HuggingFace, including:

- Transformer-like LLMs (e.g., Llama)
- Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
- Embedding Models (e.g., E5-Mistral)
- Multi-modal LLMs (e.g., LLaVA)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with `pip` or [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source):

```bash
pip install vllm
```

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## SM120 / FA3 Fork

This repository is a standalone fork for NVIDIA SM120 Blackwell workstation GPUs.
It is not intended to be merged back into the original `vllm-project/vllm` tree.

### Base

- Upstream vLLM base: `0.18.1rc1.dev125+g1bf2ddd0e`
- External FlashAttention source: `yhfgyyf/flash-attention-sm120-fa3-progress`
- External FlashAttention commit: `cee537ecf73d002dfeedc626812eeda100c16b07`

### Tested Environment

- CUDA toolkit: `13.2`
- PyTorch: `2.11.0+cu130`
- GPU architecture: `SM120`
- Tested GPU family: NVIDIA Blackwell workstation (`RTX PRO 6000 Blackwell Workstation Edition`)

### Differences From Upstream

- Enables `FA3` as the default text attention path on `SM120`.
- Bridges vLLM to the external FlashAttention 3 implementation on `SM120`.
- Keeps multimodal encoder attention on `FLASH_ATTN` and `FA3` on `SM120`.
- Fixes Qwen3-VL/Qwen3.5-VL dummy multimodal profiling so startup profiling respects `max_model_len`-bounded encoder budgets instead of always constructing maximum-size dummy videos.
- Targets CUDA `13.2` toolchain usage for FlashInfer / external FA3 builds in the provided install flow.

### Quick Install

Use an existing Python environment, preferably a fresh conda env with the
desired PyTorch already installed, then run:

```bash
git clone https://github.com/yhfgyyf/vllm-sm120-fa3-cu132.git
cd vllm-sm120-fa3-cu132
bash scripts/install_sm120_fa3_cuda132.sh
```

The install script:

- clones the external FlashAttention fork into `.vendor/`
- checks out the tested FlashAttention commit
- builds the Hopper/SM120 FA3 package with CUDA 13.2
- installs this vLLM fork in editable mode

### Important Notes

- This fork is source-first; it does not ship prebuilt `.so` binaries.
- `max_model_len` still needs to fit GPU KV-cache memory. This fork fixes the
  Qwen3-VL startup profiling path, but it does not override fundamental memory
  limits.
- For some large multimodal models, `--enforce-eager` is a practical first-step
  validation option before enabling full compile/cudagraph paths.

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
