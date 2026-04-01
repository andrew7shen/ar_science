# Unlocking LLM Creativity in Science through Analogical Reasoning

Code for the paper "Unlocking LLM Creativity in Science through Analogical Reasoning" (COLM 2026).

#### Abstract
Autonomous science promises to augment scientific discovery, particularly in complex fields like biomedicine. However, this requires AI systems that can consistently generate novel and diverse solutions to open-ended problems. We evaluate LLMs on the task of open-ended solution generation and quantify their tendency to mode collapse into low-diversity generations. To mitigate this mode collapse, we introduce analogical reasoning (AR) as a new approach to solution generation. AR generates analogies to cross-domain problems based on shared relational structure, then uses those analogies to search for novel solutions. Compared to baselines, AR discovers significantly more diverse generations (improving solution diversity metrics by 90-173\%), generates novel solutions over 50\% of the time (compared to as little as 1.6\% for baselines), and produces highly creative analogies. To validate the real-world feasibility of AR, we implement AR-generated solutions across three biomedical problems, yielding consistent quantitative gains. AR-generated approaches achieve a nearly 13-fold improvement on distributional metrics for perturbation effect prediction, infer brain region interactions with a high Spearman correlation ($\rho$=0.729) to published methods, and establish state-of-the-art performance on 2 datasets for oligonucleotide property prediction.

## Project Structure

```
ar_science/
├── src/
│   ├── main.py                 # CLI entry point
│   ├── orchestrator.py         # Workflow coordinator
│   ├── config.py               # Configuration loader
│   ├── llm_client.py           # Multi-provider LLM client
│   └── agents/
│       ├── extraction.py       # Analogy extraction
│       ├── search.py           # Solution search
│       ├── assessment.py       # Scoring, ranking & solution novelty
│       ├── baseline.py         # Baseline workflow
│       └── academic_apis.py    # Semantic Scholar, arXiv, CrossRef
├── eval/
│   ├── evaluate_on_papers.py   # Evaluation benchmark
│   ├── analogy_creativity/     # Analogy creativity
│   │   └── compare_analogies_to_ground_truth.py
│   └── generation_diversity/   # Generation diversity
│       ├── analyze_embedding_diversity.py
│       ├── compare_embedding_diversity.py
│       ├── embedding_viz_utils.py
│       ├── eval_extraction_diversity.py
│       └── metrics.py
├── ar_dataset/
│   ├── data/                   # AR Dataset 
│   │   └── dataset.json
│   └── code/                   # Dataset creation pipeline
│       ├── create_dataset.py
│       ├── discovery.py
│       ├── verification.py
│       ├── extraction.py
│       ├── difficulty.py
│       ├── schema.py
│       └── utils.py
└── case_studies/
    ├── perturbench/            # Perturbation effect prediction
    │   ├── fmm_baseline/
    │   ├── la_fmm_baseline/
    │   └── la_reproduced/
    ├── brain_interaction/      # Brain region interaction
    │   ├── coupling_model_implementation/
    │   └── pcmci_native_implementation/
    └── oligogym/               # Oligonucleotide property prediction
        └── pst_tapered_eval/
```

## License

MIT
