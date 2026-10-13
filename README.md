# Across the Ages: A Multi-Agent LLM Workflow for Bridging Intergenerational Communication

## Abstract

Intergenerational communication fosters mutual understanding and emotional connection, yet divergent styles and values create persistent gaps. While LLMs have been proposed as mediators, they lack validated mechanisms for mitigating these gaps and struggle with subtle intergenerational social norms. We introduce a multi-agent facilitator workflow that detects and mitigates tensions in real time, grounding analysis in five diagnostic dimensions: intent, emotion, dialogue acts, interruptions, and six core generational gap indicators. We evaluate the framework using computational metrics across eight LLMs, complemented by rubric-based quality assessments from four human judges and eight LLM judges. Results demonstrate the workflow's effectiveness across diverse model settings, offering insights into LLM capabilities for intergenerational communication facilitation. To our knowledge, this work provides the first systematic benchmark for LLM-based intergenerational dialogue facilitation and establishes a foundation for AI-mediated family communication support.

## Experimental Conditions

We evaluate three conditions:

1. **Baseline:** Original human conversations without simulation or facilitation.
2. **Simulation Without Facilitation:** Conversations re-simulated by an LLM without facilitator-generated social cues or facilitation.
3. **Simulation With Facilitation:** Conversations re-simulated using the multi-agent facilitator workflow, which analyzes social cues and provides facilitation.

## Repository Structure

| Directory / File | Description |
|---|---|
| `conditions/baseline_without_facilitator/` | Original human conversations used as the baseline condition |
| `conditions/sim_no_label_condition/` | Simulated conversations without facilitator-generated social cues |
| `conditions/with_facilitator/` | Simulated conversations with the multi-agent facilitator |
| `Simulation workflow/sim_no_label.py` | Implementation of the simulation workflow without facilitation |
| `Simulation workflow/Facilitator_workflow.py` | Implementation of the simulation workflow with the multi-agent facilitator |
| `dataset/` | Dialogue datasets used in the study |
| `research_guidelines/` | Annotation guidelines used for human evaluations |
| `ablation_study/` | Ablation study results and analysis scripts |
