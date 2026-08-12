---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:8456
- loss:BinaryCrossEntropyLoss
pipeline_tag: text-ranking
library_name: sentence-transformers
metrics:
- map
- mrr@10
- ndcg@10
model-index:
- name: CrossEncoder
  results:
  - task:
      type: cross-encoder-reranking
      name: Cross Encoder Reranking
    dataset:
      name: stage3val
      type: stage3val
    metrics:
    - type: map
      value: 0.939729490280961
      name: Map
    - type: mrr@10
      value: 0.9467203548085901
      name: Mrr@10
    - type: ndcg@10
      value: 0.9550020580435629
      name: Ndcg@10
---

# CrossEncoder

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model trained using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'BertForSequenceClassification'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ['Pysa; Ryuk; ransomware — Data from Intel 471 shows this pattern following attacks carried out with popular ransomware variants, such as DoppelPaymer, Maze, Netwalker, Ryuk and REvil, as well as lesser-known variants like LockBit, Nefilim, Pysa and Thanos.', 'S1073 — Royal: [Royal](https://attack.mitre.org/software/S1073) is ransomware that first appeared in early 2022; a version that also targets ESXi servers was later observed in February 2023. [Royal](https://attack.mitre.org/software/S1073) employs partial encryption and multiple threads to evade detection and speed encryption. [Royal](https://attack.mitre.org/software/S1073) has been used in attacks against multiple industries worldwide--including critical infrastructure. Security researchers have identified similarities in the encryption routines and TTPs used in [Royal](https://attack.mitre.org/software/S1073) and [Conti](https://attack.mitre.org/software/S0575) attacks and noted a possible connection between their operators.'],
    ['port 535 and with HTTP requests — When looking for which domains have resolved to this IP address it is observed that the only one categorized as malware is related to spdns.de and searching for this domain name we come to the analysis <https://gist.github.com/JohnLaTwC/ccdcbeb85649ef9feaae045482d694b9>(from @ JohnLaTwC) that shows how this domain is configured with port 535 and with HTTP requests from RAT Houdini.', 'T1584.001 — Domains: Adversaries may hijack domains and/or subdomains that can be used during targeting. Domain registration hijacking is the act of changing the registration of a domain name without the permission of the original registrant. Adversaries may gain access to an email account for the person listed as the owner of the domain. The adversary can then claim that they forgot their password in order to make changes to the domain registration. Other possibilities include social engineering a domain registration help desk to gain access to an account or taking advantage of renewal process gaps. Subdomain hijacking can occur when organizations have DNS entries that point to non-existent or deprovisioned resources. In such cases, an adversary may take control of a subdomain to conduct operations with the benefit of the trust associated with that domain. Adversaries who compromise a domain may also engage in domain shadowing by creating malicious subdomains under their control while keeping any...'],
    ['DNS over HTTPs (DoH — * The attack used state-of-the-art techniques, such as DNS over HTTPs (DoH )', 'T1110.003 — Password Spraying: Adversaries may use a single or small list of commonly used passwords against many different accounts to attempt to acquire valid account credentials. Password spraying uses one password (e.g. \'Password01\'), or a small list of commonly used passwords, that may match the complexity policy of the domain. Logins are attempted with that password against many different accounts on a network to avoid account lockouts that would normally occur when brute forcing a single account with many passwords. Typically, management services over commonly used ports are used when password spraying. Commonly targeted services include the following: * SSH (22/TCP) * Telnet (23/TCP) * FTP (21/TCP) * NetBIOS / SMB / Samba (139/TCP & 445/TCP) * LDAP (389/TCP) * Kerberos (88/TCP) * RDP / Terminal Services (3389/TCP) * HTTP/HTTP Management Services (80/TCP & 443/TCP) * MSSQL (1433/TCP) * Oracle (1521/TCP) * MySQL (3306/TCP) * VNC (5900/TCP) In addition to management services, adversaries may "target single...'],
    ['C2 — | http://999080321yomtest251-service10020125999080321[.]ru/ | URL | C2 |', 'T1102 — Web Service: Adversaries may use an existing, legitimate external Web service as a means for relaying data to/from a compromised system. Popular websites and social media acting as a mechanism for C2 may give a significant amount of cover due to the likelihood that hosts within a network are already communicating with them prior to a compromise. Using common services, such as those offered by Google or Twitter, makes it easier for adversaries to hide in expected noise. Web service providers commonly use SSL/TLS encryption, giving adversaries an added level of protection. Use of Web services may also protect back-end C2 infrastructure from discovery through malware binary analysis while also enabling operational resiliency (since this infrastructure may be dynamically changed).'],
    ['*runtime\\_report.dll** is decrypted and mapped as **TinyMet.dll** — * *runtime\\_report.dll** is decrypted and mapped as **TinyMet.dll** .', 'T1003.007 — Proc Filesystem: Adversaries may gather credentials from the proc filesystem or `/proc`. The proc filesystem is a pseudo-filesystem used as an interface to kernel data structures for Linux based systems managing virtual memory. For each process, the `/proc/<PID>/maps` file shows how memory is mapped within the process’s virtual address space. And `/proc/<PID>/mem`, exposed for debugging purposes, provides access to the process’s virtual address space. When executing with root privileges, adversaries can search these memory locations for all processes on a system that contain patterns that are indicative of credentials, such as looking for fixed strings in memory structures or cached hashes. When running without privileged access, processes can still view their own virtual memory locations. Some services or programs may save credentials in clear text inside the process’s memory. If running as or with the permissions of a web browser, a process can search the `/maps` & `/mem` locations for common...'],
]
scores = model.predict(pairs)
print(scores)
# [-3.6811 -4.3414 -4.9635 -0.2727 -4.4837]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Pysa; Ryuk; ransomware — Data from Intel 471 shows this pattern following attacks carried out with popular ransomware variants, such as DoppelPaymer, Maze, Netwalker, Ryuk and REvil, as well as lesser-known variants like LockBit, Nefilim, Pysa and Thanos.',
    [
        'S1073 — Royal: [Royal](https://attack.mitre.org/software/S1073) is ransomware that first appeared in early 2022; a version that also targets ESXi servers was later observed in February 2023. [Royal](https://attack.mitre.org/software/S1073) employs partial encryption and multiple threads to evade detection and speed encryption. [Royal](https://attack.mitre.org/software/S1073) has been used in attacks against multiple industries worldwide--including critical infrastructure. Security researchers have identified similarities in the encryption routines and TTPs used in [Royal](https://attack.mitre.org/software/S1073) and [Conti](https://attack.mitre.org/software/S0575) attacks and noted a possible connection between their operators.',
        'T1584.001 — Domains: Adversaries may hijack domains and/or subdomains that can be used during targeting. Domain registration hijacking is the act of changing the registration of a domain name without the permission of the original registrant. Adversaries may gain access to an email account for the person listed as the owner of the domain. The adversary can then claim that they forgot their password in order to make changes to the domain registration. Other possibilities include social engineering a domain registration help desk to gain access to an account or taking advantage of renewal process gaps. Subdomain hijacking can occur when organizations have DNS entries that point to non-existent or deprovisioned resources. In such cases, an adversary may take control of a subdomain to conduct operations with the benefit of the trust associated with that domain. Adversaries who compromise a domain may also engage in domain shadowing by creating malicious subdomains under their control while keeping any...',
        'T1110.003 — Password Spraying: Adversaries may use a single or small list of commonly used passwords against many different accounts to attempt to acquire valid account credentials. Password spraying uses one password (e.g. \'Password01\'), or a small list of commonly used passwords, that may match the complexity policy of the domain. Logins are attempted with that password against many different accounts on a network to avoid account lockouts that would normally occur when brute forcing a single account with many passwords. Typically, management services over commonly used ports are used when password spraying. Commonly targeted services include the following: * SSH (22/TCP) * Telnet (23/TCP) * FTP (21/TCP) * NetBIOS / SMB / Samba (139/TCP & 445/TCP) * LDAP (389/TCP) * Kerberos (88/TCP) * RDP / Terminal Services (3389/TCP) * HTTP/HTTP Management Services (80/TCP & 443/TCP) * MSSQL (1433/TCP) * Oracle (1521/TCP) * MySQL (3306/TCP) * VNC (5900/TCP) In addition to management services, adversaries may "target single...',
        'T1102 — Web Service: Adversaries may use an existing, legitimate external Web service as a means for relaying data to/from a compromised system. Popular websites and social media acting as a mechanism for C2 may give a significant amount of cover due to the likelihood that hosts within a network are already communicating with them prior to a compromise. Using common services, such as those offered by Google or Twitter, makes it easier for adversaries to hide in expected noise. Web service providers commonly use SSL/TLS encryption, giving adversaries an added level of protection. Use of Web services may also protect back-end C2 infrastructure from discovery through malware binary analysis while also enabling operational resiliency (since this infrastructure may be dynamically changed).',
        'T1003.007 — Proc Filesystem: Adversaries may gather credentials from the proc filesystem or `/proc`. The proc filesystem is a pseudo-filesystem used as an interface to kernel data structures for Linux based systems managing virtual memory. For each process, the `/proc/<PID>/maps` file shows how memory is mapped within the process’s virtual address space. And `/proc/<PID>/mem`, exposed for debugging purposes, provides access to the process’s virtual address space. When executing with root privileges, adversaries can search these memory locations for all processes on a system that contain patterns that are indicative of credentials, such as looking for fixed strings in memory structures or cached hashes. When running without privileged access, processes can still view their own virtual memory locations. Some services or programs may save credentials in clear text inside the process’s memory. If running as or with the permissions of a web browser, a process can search the `/maps` & `/mem` locations for common...',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Cross Encoder Reranking

* Dataset: `stage3val`
* Evaluated with [<code>CERerankingEvaluator</code>](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CERerankingEvaluator) with these parameters:
  ```json
  {
      "at_k": 10
  }
  ```

| Metric      | Value     |
|:------------|:----------|
| map         | 0.9397    |
| mrr@10      | 0.9467    |
| **ndcg@10** | **0.955** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 8,456 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                         | sentence_1                                                                           | label                                                          |
  |:---------|:-----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type     | string                                                                             | string                                                                               | float                                                          |
  | modality | text                                                                               | text                                                                                 |                                                                |
  | details  | <ul><li>min: 8 tokens</li><li>mean: 41.96 tokens</li><li>max: 172 tokens</li></ul> | <ul><li>min: 48 tokens</li><li>mean: 172.19 tokens</li><li>max: 317 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.46</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                      | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | label            |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Pysa; Ryuk; ransomware — Data from Intel 471 shows this pattern following attacks carried out with popular ransomware variants, such as DoppelPaymer, Maze, Netwalker, Ryuk and REvil, as well as lesser-known variants like LockBit, Nefilim, Pysa and Thanos.</code>                                                                                                                                                                    | <code>S1073 — Royal: [Royal](https://attack.mitre.org/software/S1073) is ransomware that first appeared in early 2022; a version that also targets ESXi servers was later observed in February 2023. [Royal](https://attack.mitre.org/software/S1073) employs partial encryption and multiple threads to evade detection and speed encryption. [Royal](https://attack.mitre.org/software/S1073) has been used in attacks against multiple industries worldwide--including critical infrastructure. Security researchers have identified similarities in the encryption routines and TTPs used in [Royal](https://attack.mitre.org/software/S1073) and [Conti](https://attack.mitre.org/software/S0575) attacks and noted a possible connection between their operators.</code>                                                                                                                                                                                                                                                                           | <code>0.0</code> |
  | <code>port 535 and with HTTP requests — When looking for which domains have resolved to this IP address it is observed that the only one categorized as malware is related to spdns.de and searching for this domain name we come to the analysis <https://gist.github.com/JohnLaTwC/ccdcbeb85649ef9feaae045482d694b9>(from @ JohnLaTwC) that shows how this domain is configured with port 535 and with HTTP requests from RAT Houdini.</code> | <code>T1584.001 — Domains: Adversaries may hijack domains and/or subdomains that can be used during targeting. Domain registration hijacking is the act of changing the registration of a domain name without the permission of the original registrant. Adversaries may gain access to an email account for the person listed as the owner of the domain. The adversary can then claim that they forgot their password in order to make changes to the domain registration. Other possibilities include social engineering a domain registration help desk to gain access to an account or taking advantage of renewal process gaps. Subdomain hijacking can occur when organizations have DNS entries that point to non-existent or deprovisioned resources. In such cases, an adversary may take control of a subdomain to conduct operations with the benefit of the trust associated with that domain. Adversaries who compromise a domain may also engage in domain shadowing by creating malicious subdomains under their control while...</code> | <code>0.0</code> |
  | <code>DNS over HTTPs (DoH — * The attack used state-of-the-art techniques, such as DNS over HTTPs (DoH )</code>                                                                                                                                                                                                                                                                                                                                 | <code>T1110.003 — Password Spraying: Adversaries may use a single or small list of commonly used passwords against many different accounts to attempt to acquire valid account credentials. Password spraying uses one password (e.g. 'Password01'), or a small list of commonly used passwords, that may match the complexity policy of the domain. Logins are attempted with that password against many different accounts on a network to avoid account lockouts that would normally occur when brute forcing a single account with many passwords. Typically, management services over commonly used ports are used when password spraying. Commonly targeted services include the following: * SSH (22/TCP) * Telnet (23/TCP) * FTP (21/TCP) * NetBIOS / SMB / Samba (139/TCP & 445/TCP) * LDAP (389/TCP) * Kerberos (88/TCP) * RDP / Terminal Services (3389/TCP) * HTTP/HTTP Management Services (80/TCP & 443/TCP) * MSSQL (1433/TCP) * Oracle (1521/TCP) * MySQL (3306/TCP) * VNC (5900/TCP) In addition to management services, adve...</code> | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 1
- `per_device_eval_batch_size`: 16

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 1
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 16
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss | stage3val_ndcg@10 |
|:------:|:----:|:-------------:|:-----------------:|
| 0.4991 | 264  | -             | 0.9354            |
| 0.9452 | 500  | 0.3045        | -                 |
| 0.9981 | 528  | -             | 0.9550            |


### Training Time
- **Training**: 1.7 minutes
- **Evaluation**: 8.6 seconds
- **Total**: 1.8 minutes

### Framework Versions
- Python: 3.12.13
- Sentence Transformers: 5.6.0
- Transformers: 5.13.1
- PyTorch: 2.11.0+cu128
- Accelerate: 1.14.0
- Datasets: 4.0.0
- Tokenizers: 0.22.2

## Additional Resources

- [Training and Finetuning Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-reranker): the end-to-end guide for training or finetuning Cross Encoder (reranker) models.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video reranker models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): training multimodal Cross Encoders.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->