# Walk the Talk (WTT)

**WTT-lite: PubMedQA 기반 LLM 신뢰성(faithfulness) 평가**

CAU "Intro to GenAI" 수업 프로젝트. PubMedQA 데이터셋을 이용해, LLM이 생성한 답변(및 설명)이 실제로 주어진 근거 컨텍스트(evidence context)에 근거하고 있는지, 아니면 모델이 이미 알고 있는 파라메트릭 지식(prior knowledge)에 기대어 답한 것인지를 counterfactual context 기반으로 검증하는 "WTT-lite" 파이프라인을 구현한다.

## 개념: WTT-lite란

핵심 아이디어는 "모델이 설명에서 언급한 근거가, 실제로 그 근거를 지웠을 때 답이 바뀌는 근거와 일치하는가"를 측정하는 것이다.

1. **Original answer**: target LLM에게 PubMedQA 질문 + 원본 abstract context를 RAG 프롬프트로 주고 `yes`/`no`/`maybe` 답변과 설명(explanation)을 받는다.
2. **Concept extraction**: aux LLM이 위 질문/컨텍스트/답변/설명을 보고, 답에 영향을 줄 수 있는 biomedical concept(수치, 질환, 비교군, 결과, 통계적 유의성, 연구 설계, 추적 기간 등)을 최대 N개 추출한다. 각 concept은 `current_value` → `alternative_value`(counterfactual 값)를 함께 갖는다.
3. **Counterfactual context generation**: aux LLM이 원본 context에서 해당 concept 하나만 최소한으로 바꾼 counterfactual context를 생성한다(다른 내용은 최대한 그대로 유지).
4. **Counterfactual answer**: 같은 target LLM에게 동일한 질문 + counterfactual context를 다시 주고 답변을 받는다.
5. **Scoring**: 각 concept에 대해 "모델이 설명에서 이 concept을 언급했는가(mentioned)"와 "이 concept을 실제로 바꿨을 때 답이 바뀌었는가(influential)"를 비교한다.
   - `walk_the_talk_lite_score` = `|mentioned ∩ influential| / |mentioned ∪ influential|` (샘플 단위 Jaccard 유사도)
   - 점수가 높을수록 모델이 "말한 근거"와 "실제로 의존한 근거"가 일치 = faithful. 언급했지만 실제로는 영향이 없는 concept은 `overclaimed_concepts`, 언급하지 않았지만 실제로 영향을 준 concept은 `hidden_influential_concepts`로 별도 기록된다.

## 실행 방법

메인 노트북은 **`0521_intro2genai_demo_batched.ipynb`** 하나다. Google Colab에서 실행하도록 설계되어 있으며, 로컬 LLM은 Ollama로, 일부 모델(`google/...`, `gemini...`로 시작하는 이름)은 `google.colab.ai`로 라우팅한다(`llm_generate`가 모델 이름에 따라 자동 분기).

노트북 셀 구성 (순서대로):

- **Colab setup**: `datasets`, `pandas`, `tqdm`, `matplotlib`, `requests`, `ollama`, `pydantic` 설치, `zstd` 설치, Ollama 서버 설치 스크립트(`curl -fsSL https://ollama.com/install.sh | sh`) 실행.
- **Experiment Code**: pydantic 스키마 정의(`PubMedQAAnswerSchema`, `ConceptExtractionSchema`, `CounterfactualContextSchema` 등) + 파이프라인 본체(데이터 로딩, LLM 호출, concept 추출, counterfactual 생성, 채점, 리포트/플롯 생성까지 전부 포함된 단일 셀).
- **Run**: `run_colab_experiment(...)` 호출로 실제 배치 실험 실행.
- **Inspect Outputs**: 결과 CSV 미리보기(`ALL__concept_table.csv`, `ALL__sample_summary.csv`).
- 결과 폴더를 zip으로 압축(`pubmedqa_wtt_batch_cache.zip`) 후 `google.colab.files.download`로 다운로드, 마지막에 `runtime.unassign()`으로 Colab 런타임 종료.

### 주요 설정 값

- `TARGET_MODELS`: 실제로 질문에 답하는(그리고 counterfactual context에도 다시 답하는) 모델 목록. Ollama 모델(`qwen2.5:72b`, `qwen3.6:27b`, `gemma4:31b`, `llama3.1:70b`, `gpt-oss:20b` 등) 또는 Colab AI 모델(`google/gemini-2.5-flash` 등)을 자유롭게 조합 가능.
- `AUX_MODELS`: concept 추출 + counterfactual context 생성을 담당하는 보조 모델 목록(코드상 기본값은 `gemma4:e4b` 하나).
- `TARGET_MODELS × AUX_MODELS`의 모든 조합이 하나의 "run"(`run_name = target_{...}__aux_{...}`)이 되어 순차 실행된다.
- `N_SAMPLES`, `MAX_CONTEXT_CHARS`, `MAX_CONCEPTS_PER_SAMPLE`, `OVERWRITE_CACHE` 등을 `run_colab_experiment()` 호출 인자로 조절.
- Ollama 관련: `AUTO_START_OLLAMA`, `AUTO_PULL_MODELS`, `AUTO_REMOVE_TARGET_MODELS_ON_DISK_PRESSURE`(디스크 부족 시 사용하지 않는 target 모델을 자동 삭제, aux 모델은 `PROTECTED_OLLAMA_MODELS`로 보호), `OLLAMA_KEEP_ALIVE` 등 Colab의 제한된 디스크/메모리 환경에서 여러 대형 모델을 순환시키기 위한 로직이 포함되어 있다.
- 파이프라인은 4단계(phase)로 나뉘어 모델별로 배치 처리된다: (1) target별 original answer, (2) aux별 concept + counterfactual context 생성, (3) target별 counterfactual answer, (4) run별 WTT 테이블 집계. 이렇게 하면 모델 로드/언로드 횟수를 최소화할 수 있다.
- 데이터는 HuggingFace `datasets`로 `qiaojin/PubMedQA`(`pqa_labeled` subset)를 로드하고, context 길이와 gold label(`yes`/`no`/`maybe`) 유무로 샘플을 필터링해서 사용한다.

## 저장소 구조

```
0521_intro2genai_demo_batched.ipynb   # 메인 실험 노트북 (위 파이프라인 전체)
pubmedqa_wtt_batch_cache 8/           # 최근 실행 결과 (직전 run)
pubmedqa_wtt_batch_cache 9/           # 최근 실행 결과 (가장 최신 run, excluded_samples 포함)
combined_cache_7_8_9__*.csv           # 레거시: cache 7/8/9를 합친 분석용 산출물
LLM_Faithfulness_Comparison*.png      # 레거시: 이전 분석 노트북이 생성한 그래프
target_*__aux_*__*.jsonl              # 레거시: 이전 분석 과정에서 남은 낱개 jsonl
.deepeval/                            # deepeval 패키지의 로컬 텔레메트리/캐시 디렉터리
```

`pubmedqa_wtt_batch_cache 8`과 `9`는 각각 `TARGET_MODELS × AUX_MODELS` 조합(run)마다 아래 파일을 갖는다. cache 9는 cache 8보다 더 많은 target 모델 조합을 포함하고, 각 run마다 `excluded_samples.csv`가 추가로 생성된 더 최신 실행 결과다:

| 파일 | 내용 |
|---|---|
| `{run_name}__original_answers.jsonl` | 원본 context에 대한 target 모델의 답변. 필드: `run_name, sample_idx, sample_id, question, gold_label, target_model, answer_label, answer, explanation, error, raw` |
| `{run_name}__aux_counterfactuals.jsonl` | aux 모델이 추출한 concept + 생성한 counterfactual context. 필드: `run_name, sample_idx, sample_id, concept_i, concept, category, description, current_value, alternative_value, mentioned_in_explanation, counterfactual_context, changed, change_summary, changed_span_original, changed_span_counterfactual, cf_generation_warnings, aux_model, raw_aux_output, raw_cf_output` |
| `{run_name}__counterfactual_answers.jsonl` | counterfactual context에 대한 target 모델의 답변. 필드: `run_name, sample_idx, sample_id, concept_i, concept, target_model, cf_answer_label, cf_answer, cf_explanation, cf_error, cf_raw` |
| `{run_name}__concept_table.csv` | 위 세 파일을 concept 단위로 조인한 테이블(원본 답변 vs counterfactual 답변, `answer_changed` 등 포함) |
| `{run_name}__sample_summary.csv` | 샘플(질문) 단위 집계. `num_mentioned`, `num_influential`, `intersection`, `hidden_influential_concepts`, `overclaimed_concepts`, `walk_the_talk_lite_score` 등 |
| `{run_name}__excluded_samples.csv` (cache 9만 해당) | 채점에서 제외된 샘플과 제외 사유(`reasons`) 기록 |
| `ALL__concept_table.csv` / `ALL__sample_summary.csv` / `ALL__run_summary.csv` | 폴더 내 모든 run을 합친 concept/sample/run 단위 집계 |
| `ALL__run_summary_plot.png` | run별 요약 지표(`mean_wtt_score` 등) 막대그래프 |

## 레거시 루트 파일

`combined_cache_7_8_9__*.csv`, `LLM_Faithfulness_Comparison*.png`, 루트에 낱개로 남아있는 `target_*__aux_*__*.jsonl` 파일들은 이전 분석 패스에서 생성된 결과물이다. 이를 생성했던 노트북(`llm_faithfulness_comparison.ipynb`, `llm_faithfulness_comparison_cache_7_8_9.ipynb` 등)은 이후 정리되어 저장소에서 삭제되었지만, 산출물 자체는 참고용으로 남겨두었다. 재현하려면 `0521_intro2genai_demo_batched.ipynb`를 다시 실행해서 새 캐시 폴더를 생성해야 한다.
