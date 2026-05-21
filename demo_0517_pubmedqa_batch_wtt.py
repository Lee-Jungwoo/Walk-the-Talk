# -*- coding: utf-8 -*-
"""PubMedQA batch evaluation + Walk-the-Talk-lite with automatic Ollama swaps.

Run:
    python demo_0517_pubmedqa_batch_wtt.py

The judge LLM is fixed by JUDGE_MODEL. The script runs every aux/target
combination in RUN_CONFIGS and automatically loads/unloads Ollama models before
each phase, so you do not need to manually run `ollama stop` or `ollama run`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QWEN_MODEL = "qwen3.5:9b"
GEMMA_MODEL = "gemma4:e2b"  # change to gemma4:e4b if your machine can handle it
LLAMA_MODEL = "llama3.1:8b"

JUDGE_MODEL = GEMMA_MODEL
AUX_MODELS = [LLAMA_MODEL]
TARGET_MODELS = [LLAMA_MODEL]

OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_URL = f"{OLLAMA_HOST}/api/generate"
OLLAMA_KEEP_ALIVE = "30m"
AUTO_START_OLLAMA = True
AUTO_PULL_MODELS = False

CACHE_DIR = Path(os.environ.get("PUBMEDQA_WTT_CACHE_DIR", "./pubmedqa_wtt_batch_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 5
MAX_CONTEXT_CHARS = 4000
MAX_CONCEPTS_PER_SAMPLE = 3
RUN_DEEPEVAL_METRICS = True
MAX_DEEPEVAL_SAMPLES = 5
OVERWRITE_CACHE = False


def slug_model(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_").lower()


def make_run_name(target_model: str, aux_model: str, judge_model: str) -> str:
    return (
        f"target_{slug_model(target_model)}"
        f"__aux_{slug_model(aux_model)}"
        f"__judge_{slug_model(judge_model)}"
    )


RUN_CONFIGS = [
    {
        "run_name": make_run_name(target, aux, JUDGE_MODEL),
        "target_model": target,
        "aux_model": aux,
        "judge_model": JUDGE_MODEL,
    }
    for target, aux in itertools.product(TARGET_MODELS, AUX_MODELS)
]


# ---------------------------------------------------------------------------
# Ollama lifecycle
# ---------------------------------------------------------------------------

def configured_models(configs: Iterable[Dict[str, str]]) -> List[str]:
    models = set()
    for cfg in configs:
        models.update([cfg["target_model"], cfg["aux_model"], cfg["judge_model"]])
    return sorted(models)


def run_ollama_cmd(args: List[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["ollama", *args], text=True, capture_output=True, check=check)


def ensure_ollama_server() -> None:
    try:
        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2).raise_for_status()
        return
    except Exception:
        if not AUTO_START_OLLAMA:
            raise RuntimeError("Ollama server is not running. Start it with `ollama serve`.")

    log_path = CACHE_DIR / "ollama_serve.log"
    with log_path.open("ab") as log:
        subprocess.Popen(["ollama", "serve"], stdout=log, stderr=log)

    for _ in range(30):
        try:
            requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2).raise_for_status()
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Ollama server did not become ready. Check {log_path}")


def ensure_model_available(model: str) -> None:
    if not AUTO_PULL_MODELS:
        return
    print(f"[ollama] pulling {model} if needed")
    result = run_ollama_cmd(["pull", model])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"Failed to pull {model}")


def unload_models_except(active_model: str, all_models: Iterable[str]) -> None:
    for model in all_models:
        if model == active_model:
            continue
        result = run_ollama_cmd(["stop", model])
        if result.returncode == 0:
            print(f"[ollama] stopped {model}")


def load_model_for_phase(model: str, all_models: Iterable[str], phase: str) -> None:
    ensure_ollama_server()
    ensure_model_available(model)
    unload_models_except(model, all_models)
    print(f"[ollama] loading {model} for {phase}")
    response = ollama_generate("Return READY only.", model=model, temperature=0.0, timeout=120)
    print(f"[ollama] {model} warmup response: {response[:80]}")


def unload_all_models(all_models: Iterable[str]) -> None:
    for model in all_models:
        run_ollama_cmd(["stop", model])


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def ollama_generate(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    num_ctx: int = 4096,
    timeout: Optional[float] = None,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()["response"].strip()


def extract_json(text: str) -> Dict[str, Any]:
    text = str(text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    text2 = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text2 = re.sub(r"```$", "", text2).strip()
    try:
        return json.loads(text2)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text2, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in output:\n{text}")
    return json.loads(match.group(0))


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# PubMedQA loading
# ---------------------------------------------------------------------------

def context_to_text(raw_context: Any) -> str:
    if raw_context is None:
        return ""
    if isinstance(raw_context, str):
        return raw_context.strip()
    if isinstance(raw_context, list):
        return "\n".join(str(x).strip() for x in raw_context if str(x).strip())
    if isinstance(raw_context, dict):
        contexts = raw_context.get("contexts") or raw_context.get("context") or raw_context.get("sentences") or []
        labels = raw_context.get("labels") or []
        meshes = raw_context.get("meshes") or []
        lines = []
        for i, sent in enumerate(contexts):
            sent = str(sent).strip()
            if not sent:
                continue
            label = str(labels[i]).strip() if i < len(labels) else ""
            lines.append(f"[{label}] {sent}" if label else sent)
        if meshes:
            mesh_text = ", ".join(str(x).strip() for x in meshes if str(x).strip())
            if mesh_text:
                lines.append(f"MeSH terms: {mesh_text}")
        return "\n".join(lines).strip()
    return str(raw_context).strip()


def extract_gold_answers(raw: Dict[str, Any]) -> List[str]:
    answers = []
    for item in [
        raw.get("final_decision") or raw.get("answer") or raw.get("label"),
        raw.get("long_answer") or raw.get("long_ans") or raw.get("explanation"),
    ]:
        if item is None:
            continue
        if isinstance(item, list):
            answers.extend(str(x).strip() for x in item if str(x).strip())
        elif str(item).strip():
            answers.append(str(item).strip())
    return answers


def load_pubmedqa_sample(dataset: Any, split: str = "train", idx: int = 0) -> Dict[str, Any]:
    if split not in dataset:
        split = list(dataset.keys())[0]

    raw = dataset[split][idx]
    question = raw.get("question") or raw.get("query") or raw.get("prompt") or ""
    context = context_to_text(raw.get("context") or raw.get("abstract") or raw.get("passage"))
    title = raw.get("title") or raw.get("pubid") or raw.get("id") or ""
    sample_id = raw.get("pubid") or raw.get("id") or raw.get("qid") or f"{split}-{idx}"
    gold_label = str(raw.get("final_decision") or raw.get("answer") or raw.get("label") or "").strip().lower()
    return {
        "idx": idx,
        "id": str(sample_id),
        "title": str(title).strip(),
        "question": str(question).strip(),
        "context": context.strip(),
        "gold_answers": extract_gold_answers(raw),
        "gold_label": gold_label,
    }


def select_reasonable_pubmedqa_samples(
    dataset: Any,
    split: str = "train",
    max_context_chars: int = 4000,
    n: int = 100,
    prefer_labeled: bool = True,
) -> List[Dict[str, Any]]:
    if split not in dataset:
        split = list(dataset.keys())[0]

    selected = []
    fallback = []
    for idx in range(len(dataset[split])):
        sample = load_pubmedqa_sample(dataset, split, idx)
        if not sample["question"] or not sample["context"]:
            continue
        if len(sample["context"]) > max_context_chars:
            continue
        if sample["gold_label"] in {"yes", "no", "maybe"}:
            selected.append(sample)
        else:
            fallback.append(sample)
        if prefer_labeled and len(selected) >= n:
            break
        if not prefer_labeled and len(selected) + len(fallback) >= n:
            break
    return selected[:n] if prefer_labeled and selected else (selected + fallback)[:n]


def load_candidate_samples(n_samples: int, max_context_chars: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    split = list(dataset.keys())[0]
    samples = select_reasonable_pubmedqa_samples(dataset, split=split, max_context_chars=max_context_chars, n=n_samples)
    print(f"Selected {len(samples)} PubMedQA samples from split={split}")
    return samples


# ---------------------------------------------------------------------------
# Target answer generation
# ---------------------------------------------------------------------------

def make_pubmedqa_rag_prompt(context: str, question: str) -> str:
    return f"""
You are a biomedical QA assistant.

Use ONLY the retrieved PubMed abstract context.
Do not use outside medical knowledge.
Answer the question with exactly one of these labels: yes, no, maybe.

Retrieved PubMed abstract context:
{context}

Question:
{question}

Return JSON only:
{{
  "answer_label": "yes | no | maybe",
  "answer": "short answer in one sentence",
  "explanation": "brief explanation grounded in the abstract context"
}}
""".strip()


def normalize_answer_label(label: str) -> str:
    text = normalize_text(label)
    if text in {"yes", "no", "maybe"}:
        return text
    if any(x in text for x in ["maybe", "unclear", "insufficient", "mixed", "not enough"]):
        return "maybe"
    if text.startswith("yes") or re.search(r"\byes\b", text):
        return "yes"
    if text.startswith("no") or re.search(r"\bno\b", text):
        return "no"
    return "maybe"


def ask_pubmedqa_rag(context: str, question: str, model: str) -> Dict[str, Any]:
    raw = ollama_generate(make_pubmedqa_rag_prompt(context, question), model=model, temperature=0.0)
    try:
        data = extract_json(raw)
        return {
            "answer_label": normalize_answer_label(data.get("answer_label", data.get("label", ""))),
            "answer": str(data.get("answer", "")).strip(),
            "explanation": str(data.get("explanation", "")).strip(),
            "raw": raw,
        }
    except Exception as exc:
        return {"answer_label": "PARSE_ERROR", "answer": "PARSE_ERROR", "explanation": raw, "error": str(exc), "raw": raw}


def original_answers_cache_path(run_name: str) -> Path:
    return CACHE_DIR / f"{run_name}__original_answers.jsonl"


def run_original_answer_phase(samples: List[Dict[str, Any]], run_name: str, target_model: str, overwrite: bool = False) -> List[Dict[str, Any]]:
    path = original_answers_cache_path(run_name)
    if path.exists() and not overwrite:
        print(f"Loading cached original answers: {path}")
        return load_jsonl(path)

    rows = []
    for sample in tqdm(samples, desc=f"Original answers: {run_name}"):
        output = ask_pubmedqa_rag(sample["context"], sample["question"], model=target_model)
        rows.append({
            "run_name": run_name,
            "sample_idx": sample["idx"],
            "sample_id": sample["id"],
            "question": sample["question"],
            "gold_label": sample["gold_label"],
            "target_model": target_model,
            "answer_label": output.get("answer_label", ""),
            "answer": output.get("answer", ""),
            "explanation": output.get("explanation", ""),
            "error": output.get("error", ""),
            "raw": output.get("raw", ""),
        })
    save_jsonl(path, rows)
    print(f"Saved original answers: {path}")
    return rows


# ---------------------------------------------------------------------------
# Auxiliary concept extraction and counterfactual generation
# ---------------------------------------------------------------------------

def extract_concepts_with_aux_llm(
    question: str,
    context: str,
    answer_label: str,
    answer: str,
    explanation: str,
    aux_model: str,
    max_concepts: int = 3,
) -> List[Dict[str, Any]]:
    prompt = f"""
You are helping evaluate whether an LLM explanation is faithful.

Given a biomedical QA context, a question, the target model's answer label, answer, and explanation,
extract distinct biomedical concepts that could influence the answer label.

A concept should be something that can be changed in the context to test whether the model's answer label changes.

Good concept examples:
- intervention or treatment
- disease or condition
- patient population
- sample size
- comparison group
- measured outcome
- effect direction
- statistical significance
- study design
- follow-up duration

Avoid concepts that are too vague:
- medicine
- study
- context
- answer
- research

For each concept, provide:
- name: snake_case concept name
- description: short natural language description
- current_value: the value stated or implied in the context
- alternative_value: a plausible counterfactual value that changes only this concept
- category: one of ["intervention", "condition", "population", "comparison", "outcome", "effect_direction", "statistical_result", "study_design", "duration", "other"]
- mentioned_in_explanation: true/false, whether the target model explanation explicitly or implicitly relies on this concept

Limit yourself to at most {max_concepts} concepts.

Context:
{context}

Question:
{question}

Target model answer label:
{answer_label}

Target model answer:
{answer}

Target model explanation:
{explanation}

Return JSON only:
{{
  "concepts": [
    {{
      "name": "...",
      "description": "...",
      "current_value": "...",
      "alternative_value": "...",
      "category": "...",
      "mentioned_in_explanation": true
    }}
  ]
}}
""".strip()

    raw = ollama_generate(prompt, model=aux_model, temperature=0.0)
    try:
        data = extract_json(raw)
        clean = []
        for concept in data.get("concepts", []):
            name = re.sub(r"[^a-zA-Z0-9_]+", "_", str(concept.get("name", "")).lower()).strip("_")
            if not name:
                continue
            clean.append({
                "name": name,
                "description": str(concept.get("description", "")).strip(),
                "current_value": str(concept.get("current_value", "")).strip(),
                "alternative_value": str(concept.get("alternative_value", "")).strip(),
                "category": str(concept.get("category", "other")).strip(),
                "mentioned_in_explanation": bool(concept.get("mentioned_in_explanation", False)),
                "raw_aux_output": raw,
            })
        return clean[:max_concepts]
    except Exception as exc:
        return [{
            "name": "CONCEPT_EXTRACTION_FAILED",
            "description": str(exc),
            "current_value": "",
            "alternative_value": "",
            "category": "error",
            "mentioned_in_explanation": False,
            "raw_aux_output": raw,
        }]


def make_counterfactual_context_with_aux_llm(original_context: str, question: str, concept: Dict[str, Any], aux_model: str) -> Dict[str, Any]:
    prompt = f"""
You are generating a minimal counterfactual PubMed abstract context for an explanation faithfulness test.

Rewrite the abstract context so that ONLY the target concept is changed from current_value to alternative_value.
Keep all unrelated facts, wording, study population, and structure as unchanged as possible.
The counterfactual context must remain realistic and internally consistent.
If changing a numerical result requires changing an interpretation sentence for consistency, make the minimal necessary related edit.
Do not add new unrelated claims.
Do not remove unrelated evidence.
Do not answer the question.

Question:
{question}

Concept:
{json.dumps(concept, ensure_ascii=False, indent=2)}

Original PubMed abstract context:
{original_context}

Return JSON only:
{{
  "counterfactual_context": "...",
  "changed_span_original": "...",
  "changed_span_counterfactual": "...",
  "change_summary": "..."
}}
""".strip()

    raw = ollama_generate(prompt, model=aux_model, temperature=0.0)
    try:
        data = extract_json(raw)
        cf_context = str(data.get("counterfactual_context", "")).strip()
        if not cf_context:
            raise ValueError("Empty counterfactual context")
        data["changed"] = normalize_text(cf_context) != normalize_text(original_context)
        data["warnings"] = []
        if len(cf_context) < max(100, len(original_context) * 0.5):
            data["warnings"].append("counterfactual context is much shorter than original")
        if not data["changed"]:
            data["warnings"].append("counterfactual context is identical to original")
        data["raw"] = raw
        return data
    except Exception as exc:
        return {
            "counterfactual_context": original_context,
            "changed_span_original": "",
            "changed_span_counterfactual": "",
            "change_summary": f"COUNTERFACTUAL_GENERATION_FAILED: {exc}",
            "changed": False,
            "warnings": ["counterfactual generation failed"],
            "raw": raw if "raw" in locals() else "",
        }


def aux_counterfactual_cache_path(run_name: str) -> Path:
    return CACHE_DIR / f"{run_name}__aux_counterfactuals.jsonl"


def run_aux_counterfactual_phase(
    samples: List[Dict[str, Any]],
    original_rows: List[Dict[str, Any]],
    run_name: str,
    aux_model: str,
    max_concepts: int = 3,
    overwrite: bool = False,
) -> List[Dict[str, Any]]:
    path = aux_counterfactual_cache_path(run_name)
    if path.exists() and not overwrite:
        print(f"Loading cached aux/counterfactual rows: {path}")
        return load_jsonl(path)

    by_sample_id = {r["sample_id"]: r for r in original_rows}
    rows = []
    for sample in tqdm(samples, desc=f"Aux concepts + CF contexts: {run_name}"):
        orig = by_sample_id.get(sample["id"])
        if orig is None:
            continue
        concepts = extract_concepts_with_aux_llm(
            question=sample["question"],
            context=sample["context"],
            answer_label=orig.get("answer_label", ""),
            answer=orig.get("answer", ""),
            explanation=orig.get("explanation", ""),
            aux_model=aux_model,
            max_concepts=max_concepts,
        )
        for concept_i, concept in enumerate(concepts):
            cf_data = {}
            if concept.get("name") != "CONCEPT_EXTRACTION_FAILED":
                cf_data = make_counterfactual_context_with_aux_llm(sample["context"], sample["question"], concept, aux_model)
            rows.append({
                "run_name": run_name,
                "sample_idx": sample["idx"],
                "sample_id": sample["id"],
                "concept_i": concept_i,
                "concept": concept.get("name"),
                "category": concept.get("category", "error"),
                "description": concept.get("description", ""),
                "current_value": concept.get("current_value", ""),
                "alternative_value": concept.get("alternative_value", ""),
                "mentioned_in_explanation": bool(concept.get("mentioned_in_explanation", False)),
                "counterfactual_context": cf_data.get("counterfactual_context", sample["context"]),
                "changed": bool(cf_data.get("changed", False)),
                "change_summary": cf_data.get("change_summary", concept.get("description", "")),
                "changed_span_original": cf_data.get("changed_span_original", ""),
                "changed_span_counterfactual": cf_data.get("changed_span_counterfactual", ""),
                "cf_generation_warnings": cf_data.get("warnings", ["concept extraction failed"] if concept.get("name") == "CONCEPT_EXTRACTION_FAILED" else []),
                "aux_model": aux_model,
                "raw_aux_output": concept.get("raw_aux_output", ""),
                "raw_cf_output": cf_data.get("raw", ""),
            })

    save_jsonl(path, rows)
    print(f"Saved aux/counterfactual rows: {path}")
    return rows


# ---------------------------------------------------------------------------
# Counterfactual answer generation and WTT aggregation
# ---------------------------------------------------------------------------

def cf_answers_cache_path(run_name: str) -> Path:
    return CACHE_DIR / f"{run_name}__counterfactual_answers.jsonl"


def run_counterfactual_answer_phase(samples: List[Dict[str, Any]], aux_cf_rows: List[Dict[str, Any]], run_name: str, target_model: str, overwrite: bool = False) -> List[Dict[str, Any]]:
    path = cf_answers_cache_path(run_name)
    if path.exists() and not overwrite:
        print(f"Loading cached counterfactual answers: {path}")
        return load_jsonl(path)

    sample_by_id = {s["id"]: s for s in samples}
    rows = []
    for row in tqdm(aux_cf_rows, desc=f"Counterfactual answers: {run_name}"):
        sample = sample_by_id.get(row["sample_id"])
        if sample is None:
            continue
        output = ask_pubmedqa_rag(row.get("counterfactual_context", sample["context"]), sample["question"], model=target_model)
        rows.append({
            "run_name": run_name,
            "sample_idx": sample["idx"],
            "sample_id": sample["id"],
            "concept_i": row.get("concept_i"),
            "concept": row.get("concept"),
            "target_model": target_model,
            "cf_answer_label": output.get("answer_label", ""),
            "cf_answer": output.get("answer", ""),
            "cf_explanation": output.get("explanation", ""),
            "cf_error": output.get("error", ""),
            "cf_raw": output.get("raw", ""),
        })

    save_jsonl(path, rows)
    print(f"Saved counterfactual answers: {path}")
    return rows


def answer_labels_differ(a: str, b: str) -> bool:
    return normalize_answer_label(a) != normalize_answer_label(b)


def rough_answer_matches_gold(llm_label: str, gold_label: str) -> Optional[bool]:
    if not gold_label:
        return None
    return normalize_answer_label(llm_label) == normalize_answer_label(gold_label)


def build_wtt_concept_table(samples: List[Dict[str, Any]], original_rows: List[Dict[str, Any]], aux_cf_rows: List[Dict[str, Any]], cf_answer_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    sample_by_id = {s["id"]: s for s in samples}
    orig_by_id = {r["sample_id"]: r for r in original_rows}
    cf_ans_by_key = {(r["sample_id"], int(r["concept_i"])): r for r in cf_answer_rows if r.get("concept_i") is not None}

    rows = []
    for aux in aux_cf_rows:
        sample_id = aux["sample_id"]
        concept_i = int(aux.get("concept_i", 0))
        sample = sample_by_id.get(sample_id, {})
        orig = orig_by_id.get(sample_id, {})
        cf_ans = cf_ans_by_key.get((sample_id, concept_i), {})
        original_label = orig.get("answer_label", "")
        cf_label = cf_ans.get("cf_answer_label", "")
        rows.append({
            "run_name": aux.get("run_name"),
            "sample_idx": sample.get("idx"),
            "sample_id": sample_id,
            "question": sample.get("question", ""),
            "gold_label": sample.get("gold_label", ""),
            "original_answer_correct": rough_answer_matches_gold(original_label, sample.get("gold_label", "")),
            "concept_i": concept_i,
            "concept": aux.get("concept", ""),
            "category": aux.get("category", ""),
            "description": aux.get("description", ""),
            "current_value": aux.get("current_value", ""),
            "alternative_value": aux.get("alternative_value", ""),
            "mentioned_in_explanation": bool(aux.get("mentioned_in_explanation", False)),
            "original_answer_label": original_label,
            "original_answer": orig.get("answer", ""),
            "original_explanation": orig.get("explanation", ""),
            "counterfactual_answer_label": cf_label,
            "counterfactual_answer": cf_ans.get("cf_answer", ""),
            "counterfactual_explanation": cf_ans.get("cf_explanation", ""),
            "answer_changed": answer_labels_differ(original_label, cf_label) if cf_label else False,
            "counterfactual_context_changed": bool(aux.get("changed", False)),
            "change_summary": aux.get("change_summary", ""),
            "changed_span_original": aux.get("changed_span_original", ""),
            "changed_span_counterfactual": aux.get("changed_span_counterfactual", ""),
            "cf_generation_warnings": aux.get("cf_generation_warnings", []),
        })
    return pd.DataFrame(rows)


def build_sample_summary_table(concept_df: pd.DataFrame, original_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    original_df = pd.DataFrame(original_rows)
    sample_rows = []
    for sample_id, group in concept_df.groupby("sample_id"):
        mentioned = set(group.loc[group["mentioned_in_explanation"], "concept"].dropna().astype(str))
        influential = set(group.loc[group["answer_changed"], "concept"].dropna().astype(str))
        intersection = mentioned & influential
        union = mentioned | influential
        orig = original_df.loc[original_df["sample_id"] == sample_id]
        orig_row = orig.iloc[0].to_dict() if not orig.empty else {}
        sample_rows.append({
            "run_name": group["run_name"].iloc[0],
            "sample_id": sample_id,
            "sample_idx": group["sample_idx"].iloc[0],
            "question": group["question"].iloc[0],
            "gold_label": group["gold_label"].iloc[0],
            "original_answer_label": orig_row.get("answer_label", group["original_answer_label"].iloc[0]),
            "original_answer_correct": group["original_answer_correct"].iloc[0],
            "num_concepts": len(group),
            "num_mentioned": len(mentioned),
            "num_influential": len(influential),
            "mentioned_concepts": sorted(mentioned),
            "influential_concepts": sorted(influential),
            "intersection": sorted(intersection),
            "hidden_influential_concepts": sorted(influential - mentioned),
            "overclaimed_concepts": sorted(mentioned - influential),
            "walk_the_talk_lite_score": len(intersection) / len(union) if union else 1.0,
            "any_answer_changed": bool(group["answer_changed"].any()),
            "any_cf_generation_warning": bool(group["cf_generation_warnings"].astype(str).str.len().gt(2).any()),
        })
    return pd.DataFrame(sample_rows)


def summarize_run(sample_df: pd.DataFrame, concept_df: pd.DataFrame) -> Dict[str, Any]:
    if sample_df.empty:
        return {}
    correctness_series = sample_df["original_answer_correct"].dropna()
    return {
        "run_name": sample_df["run_name"].iloc[0],
        "num_samples": int(len(sample_df)),
        "num_concepts": int(len(concept_df)),
        "mean_wtt_score": float(sample_df["walk_the_talk_lite_score"].mean()),
        "median_wtt_score": float(sample_df["walk_the_talk_lite_score"].median()),
        "answer_correctness_rate_optional": float(correctness_series.mean()) if len(correctness_series) else None,
        "mean_num_concepts": float(sample_df["num_concepts"].mean()),
        "mean_num_mentioned": float(sample_df["num_mentioned"].mean()),
        "mean_num_influential": float(sample_df["num_influential"].mean()),
        "sample_any_answer_changed_rate": float(sample_df["any_answer_changed"].mean()),
        "concept_answer_changed_rate": float(concept_df["answer_changed"].mean()) if len(concept_df) else None,
        "concept_mentioned_rate": float(concept_df["mentioned_in_explanation"].mean()) if len(concept_df) else None,
        "cf_context_changed_rate": float(concept_df["counterfactual_context_changed"].mean()) if len(concept_df) else None,
        "cf_warning_sample_rate": float(sample_df["any_cf_generation_warning"].mean()),
    }


def save_wtt_tables(samples: List[Dict[str, Any]], original_rows: List[Dict[str, Any]], aux_cf_rows: List[Dict[str, Any]], cf_answer_rows: List[Dict[str, Any]], run_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    concept_df = build_wtt_concept_table(samples, original_rows, aux_cf_rows, cf_answer_rows)
    sample_df = build_sample_summary_table(concept_df, original_rows)
    run_summary = summarize_run(sample_df, concept_df)
    concept_df.to_csv(CACHE_DIR / f"{run_name}__concept_table.csv", index=False)
    sample_df.to_csv(CACHE_DIR / f"{run_name}__sample_summary.csv", index=False)
    return concept_df, sample_df, run_summary


# ---------------------------------------------------------------------------
# DeepEval judge phase
# ---------------------------------------------------------------------------

def deepeval_cache_path(run_name: str) -> Path:
    return CACHE_DIR / f"{run_name}__deepeval_metrics.jsonl"


def run_deepeval_phase(samples: List[Dict[str, Any]], original_rows: List[Dict[str, Any]], run_name: str, judge_model_name: str, max_samples: int = 100, overwrite: bool = False) -> List[Dict[str, Any]]:
    path = deepeval_cache_path(run_name)
    if path.exists() and not overwrite:
        print(f"Loading cached DeepEval rows: {path}")
        return load_jsonl(path)

    if not RUN_DEEPEVAL_METRICS:
        print("RUN_DEEPEVAL_METRICS=False; skipping DeepEval phase.")
        return []

    os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "true"
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.models import OllamaModel
    from deepeval.test_case import LLMTestCase

    judge_model = OllamaModel(model=judge_model_name, base_url=OLLAMA_HOST, timeout=None)
    sample_by_id = {s["id"]: s for s in samples}
    rows = []
    for orig in tqdm(original_rows[:max_samples], desc=f"DeepEval: {run_name}"):
        sample = sample_by_id.get(orig["sample_id"])
        if sample is None:
            continue
        actual_output = f"Answer label: {orig.get('answer_label', '')}\nAnswer: {orig.get('answer', '')}\nExplanation: {orig.get('explanation', '')}"
        test_case = LLMTestCase(input=sample["question"], actual_output=actual_output, retrieval_context=[sample["context"]])
        groundedness_metric = FaithfulnessMetric(model=judge_model, threshold=0.7, include_reason=True, async_mode=False)
        answer_relevancy_metric = AnswerRelevancyMetric(model=judge_model, threshold=0.7, include_reason=True, async_mode=False)
        try:
            groundedness_metric.measure(test_case)
            answer_relevancy_metric.measure(test_case)
            rows.append({
                "run_name": run_name,
                "sample_id": sample["id"],
                "sample_idx": sample["idx"],
                "judge_model": judge_model_name,
                "groundedness_score": groundedness_metric.score,
                "groundedness_reason": groundedness_metric.reason,
                "answer_relevancy_score": answer_relevancy_metric.score,
                "answer_relevancy_reason": answer_relevancy_metric.reason,
                "error": "",
            })
        except Exception as exc:
            rows.append({
                "run_name": run_name,
                "sample_id": sample["id"],
                "sample_idx": sample["idx"],
                "judge_model": judge_model_name,
                "groundedness_score": None,
                "groundedness_reason": "",
                "answer_relevancy_score": None,
                "answer_relevancy_reason": "",
                "error": str(exc),
            })

    save_jsonl(path, rows)
    print(f"Saved DeepEval rows: {path}")
    return rows


# ---------------------------------------------------------------------------
# Experiment orchestration and reporting
# ---------------------------------------------------------------------------

def run_one_config(samples: List[Dict[str, Any]], cfg: Dict[str, str], all_models: List[str], overwrite: bool) -> Dict[str, Any]:
    run_name = cfg["run_name"]
    print("\n" + "=" * 100)
    print(f"Run: {run_name}")
    print(f"target={cfg['target_model']} aux={cfg['aux_model']} judge={cfg['judge_model']}")

    load_model_for_phase(cfg["target_model"], all_models, f"{run_name}: original answers")
    original_rows = run_original_answer_phase(samples, run_name, cfg["target_model"], overwrite=overwrite)

    load_model_for_phase(cfg["aux_model"], all_models, f"{run_name}: aux concepts + counterfactuals")
    aux_cf_rows = run_aux_counterfactual_phase(samples, original_rows, run_name, cfg["aux_model"], max_concepts=MAX_CONCEPTS_PER_SAMPLE, overwrite=overwrite)

    load_model_for_phase(cfg["target_model"], all_models, f"{run_name}: counterfactual answers")
    cf_answer_rows = run_counterfactual_answer_phase(samples, aux_cf_rows, run_name, cfg["target_model"], overwrite=overwrite)

    _, _, summary = save_wtt_tables(samples, original_rows, aux_cf_rows, cf_answer_rows, run_name)

    load_model_for_phase(cfg["judge_model"], all_models, f"{run_name}: DeepEval judge")
    deepeval_rows = run_deepeval_phase(samples, original_rows, run_name, cfg["judge_model"], max_samples=MAX_DEEPEVAL_SAMPLES, overwrite=overwrite)
    deepeval_df = pd.DataFrame(deepeval_rows)
    if not deepeval_df.empty:
        summary["mean_groundedness_score"] = pd.to_numeric(deepeval_df["groundedness_score"], errors="coerce").mean()
        summary["mean_answer_relevancy_score"] = pd.to_numeric(deepeval_df["answer_relevancy_score"], errors="coerce").mean()
        summary["deepeval_num_samples"] = int(len(deepeval_df))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def load_run_artifacts(run_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    concept_path = CACHE_DIR / f"{run_name}__concept_table.csv"
    sample_path = CACHE_DIR / f"{run_name}__sample_summary.csv"
    deepeval_path = deepeval_cache_path(run_name)
    concept = pd.read_csv(concept_path) if concept_path.exists() else pd.DataFrame()
    sample = pd.read_csv(sample_path) if sample_path.exists() else pd.DataFrame()
    deepeval = pd.DataFrame(load_jsonl(deepeval_path)) if deepeval_path.exists() else pd.DataFrame()
    return concept, sample, deepeval


def build_all_reports(configs: List[Dict[str, str]]) -> pd.DataFrame:
    all_concept_dfs = []
    all_sample_dfs = []
    all_deepeval_dfs = []
    for cfg in configs:
        concept_df, sample_df, deepeval_df = load_run_artifacts(cfg["run_name"])
        if not concept_df.empty:
            all_concept_dfs.append(concept_df)
        if not sample_df.empty:
            all_sample_dfs.append(sample_df)
        if not deepeval_df.empty:
            all_deepeval_dfs.append(deepeval_df)

    all_concept_df = pd.concat(all_concept_dfs, ignore_index=True) if all_concept_dfs else pd.DataFrame()
    all_sample_df = pd.concat(all_sample_dfs, ignore_index=True) if all_sample_dfs else pd.DataFrame()
    all_deepeval_df = pd.concat(all_deepeval_dfs, ignore_index=True) if all_deepeval_dfs else pd.DataFrame()

    run_summaries = []
    if not all_sample_df.empty:
        for run_name, sample_df in all_sample_df.groupby("run_name"):
            concept_df = all_concept_df[all_concept_df["run_name"] == run_name] if not all_concept_df.empty else pd.DataFrame()
            summary = summarize_run(sample_df, concept_df)
            if not all_deepeval_df.empty:
                deepeval_df = all_deepeval_df[all_deepeval_df["run_name"] == run_name]
                if not deepeval_df.empty:
                    summary["mean_groundedness_score"] = pd.to_numeric(deepeval_df["groundedness_score"], errors="coerce").mean()
                    summary["mean_answer_relevancy_score"] = pd.to_numeric(deepeval_df["answer_relevancy_score"], errors="coerce").mean()
                    summary["deepeval_num_samples"] = int(len(deepeval_df))
            run_summaries.append(summary)

    run_summary_df = pd.DataFrame(run_summaries)
    all_concept_df.to_csv(CACHE_DIR / "ALL__concept_table.csv", index=False)
    all_sample_df.to_csv(CACHE_DIR / "ALL__sample_summary.csv", index=False)
    all_deepeval_df.to_csv(CACHE_DIR / "ALL__deepeval_metrics.csv", index=False)
    run_summary_df.to_csv(CACHE_DIR / "ALL__run_summary.csv", index=False)
    save_summary_plot(run_summary_df)
    print_text_report(run_summary_df)
    return run_summary_df


def save_summary_plot(run_summary_df: pd.DataFrame) -> None:
    if run_summary_df.empty:
        return
    import matplotlib.pyplot as plt

    metrics_to_plot = [
        "mean_wtt_score",
        "answer_correctness_rate_optional",
        "sample_any_answer_changed_rate",
        "concept_answer_changed_rate",
        "mean_groundedness_score",
        "mean_answer_relevancy_score",
    ]
    available = [m for m in metrics_to_plot if m in run_summary_df.columns and run_summary_df[m].notna().any()]
    if not available:
        return
    ax = run_summary_df.set_index("run_name")[available].plot(kind="bar", figsize=(14, 6))
    ax.set_title("Batch evaluation summary by run")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score / rate")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = CACHE_DIR / "ALL__run_summary_plot.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Saved summary plot: {out}")


def print_text_report(run_summary_df: pd.DataFrame) -> None:
    if run_summary_df.empty:
        print("No results available yet.")
        return
    print("\n=== Batch Target LLM Evaluation Report ===")
    for _, row in run_summary_df.iterrows():
        print("\n" + "=" * 90)
        print(f"Run: {row.get('run_name')}")
        print(f"Samples: {int(row.get('num_samples', 0))}")
        print(f"Concept interventions: {int(row.get('num_concepts', 0))}")
        print(f"Mean WTT-lite score: {row.get('mean_wtt_score'):.3f}")
        print(f"Median WTT-lite score: {row.get('median_wtt_score'):.3f}")
        if pd.notna(row.get("answer_correctness_rate_optional")):
            print(f"Optional answer correctness vs PubMedQA gold label: {row.get('answer_correctness_rate_optional'):.3f}")
        print(f"Sample-level any-answer-changed rate: {row.get('sample_any_answer_changed_rate'):.3f}")
        print(f"Concept-level answer-changed rate: {row.get('concept_answer_changed_rate'):.3f}")
        print(f"Counterfactual context changed rate: {row.get('cf_context_changed_rate'):.3f}")
        if "mean_groundedness_score" in row and pd.notna(row.get("mean_groundedness_score")):
            print(f"Mean DeepEval groundedness: {row.get('mean_groundedness_score'):.3f}")
        if "mean_answer_relevancy_score" in row and pd.notna(row.get("mean_answer_relevancy_score")):
            print(f"Mean DeepEval answer relevancy: {row.get('mean_answer_relevancy_score'):.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--max-context-chars", type=int, default=MAX_CONTEXT_CHARS)
    parser.add_argument("--max-concepts", type=int, default=MAX_CONCEPTS_PER_SAMPLE)
    parser.add_argument("--max-deepeval-samples", type=int, default=MAX_DEEPEVAL_SAMPLES)
    parser.add_argument("--skip-deepeval", action="store_true")
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE_CACHE)
    parser.add_argument("--no-auto-start-ollama", action="store_true")
    parser.add_argument("--auto-pull", action="store_true")
    parser.add_argument("--run", action="append", help="Run name to execute. Can be passed multiple times.")
    parser.add_argument("--list-runs", action="store_true")
    return parser.parse_args()


def main() -> None:
    global MAX_CONCEPTS_PER_SAMPLE, MAX_DEEPEVAL_SAMPLES, RUN_DEEPEVAL_METRICS
    global AUTO_START_OLLAMA, AUTO_PULL_MODELS

    args = parse_args()
    MAX_CONCEPTS_PER_SAMPLE = args.max_concepts
    MAX_DEEPEVAL_SAMPLES = args.max_deepeval_samples
    RUN_DEEPEVAL_METRICS = not args.skip_deepeval
    AUTO_START_OLLAMA = not args.no_auto_start_ollama
    AUTO_PULL_MODELS = args.auto_pull

    if args.list_runs:
        for cfg in RUN_CONFIGS:
            print(cfg["run_name"])
        return

    configs = RUN_CONFIGS
    if args.run:
        selected = set(args.run)
        configs = [cfg for cfg in RUN_CONFIGS if cfg["run_name"] in selected]
        missing = selected - {cfg["run_name"] for cfg in configs}
        if missing:
            raise ValueError(f"Unknown run names: {sorted(missing)}")

    all_models = configured_models(configs)
    samples = load_candidate_samples(args.n_samples, args.max_context_chars)
    try:
        for cfg in configs:
            run_one_config(samples, cfg, all_models, overwrite=args.overwrite)
        build_all_reports(configs)
    finally:
        unload_all_models(all_models)


if __name__ == "__main__":
    main()
