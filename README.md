# Walk the Talk (WTT-lite)

PubMedQA로 LLM 답변의 faithfulness를 재는 실험. CAU "Intro to GenAI" 수업 프로젝트

모델이 답을 낼 때 정말 주어진 근거(abstract)를 보고 답하는지, 아니면 그냥 알던 지식으로 답하고 근거는 나중에 갖다 붙이는지를 counterfactual로 확인하는 실험입니다.

Matton et al.의 "Walk the Talk?" 논문의 방법론을 참고했습니다.

## 아이디어

Faithfulness Score: "설명에서 언급한 근거"와 "실제로 지웠을 때 답이 바뀌는 근거"가 얼마나 겹치는지를 측정합니다.

1. **원본 answer** — target 모델에 질문 + 원본 abstract를 주고 yes/no/maybe + 설명을 받음
2. **concept 추출** — aux 모델이 답에 영향 줄 만한 biomedical concept(수치, 질환, 비교군, 통계적 유의성 등)을 뽑고, 각각에 counterfactual 값을 붙임
3. **counterfactual context** — 그 concept 하나만 최소로 바꾼 context 생성 (나머지는 그대로)
4. **counterfactual answer** — 같은 target 모델에 바뀐 context로 다시 질문
5. **채점** — concept별로 "설명에서 언급됨(mentioned)" vs "실제로 답을 바꿈(influential)"을 비교

모델의 Faithfulness는 샘플 단위 Jaccard 유사도를 평균해서 산출합니다.

```
wtt_lite_score = |mentioned ∩ influential| / |mentioned ∪ influential|
```

높을수록 말과 실제 근거가 일치 = faithful. 언급만 하고 영향 없는 건 `overclaimed`, 언급 안 했는데 영향 준 건 `hidden_influential`로 따로 기록한다.

## 결과

`pubmedqa_wtt_batch_cache 9/`의 `ALL__run_summary.csv`와 `ALL__sample_summary.csv`를 기준으로 집계했습니다. 표의 n은 채점된 샘플 수이며, 표준편차는 샘플별 `walk_the_talk_lite_score`의 표본 표준편차(ddof=1)입니다.

| Target 모델 | Aux 모델 | n | 평균 walk_the_talk_lite_score | 표준편차 |
|---|---|---:|---:|---:|
| `deepseek-r1:32b` | `gemma4:e4b` | 19 | 0.544 | 0.392 |
| `gemma4:31b` | `gemma4:e4b` | 15 | 0.511 | 0.359 |
| `gpt-oss:20b` | `gemma4:e4b` | 20 | 0.417 | 0.427 |
| `llama3.1:70b` | `gemma4:e4b` | 19 | 0.377 | 0.293 |
| `qwen2.5:72b` | `gemma4:e4b` | 20 | 0.392 | 0.402 |
| `qwen3.6:27b` | `gemma4:e4b` | 11 | 0.318 | 0.311 |

이 점수는 설명에서 언급한 개념과 반사실 개입 시 실제로 답변을 바꾸는 개념 간의 겹침을 샘플별 Jaccard 유사도로 측정한 뒤 평균한 값입니다. 각 run의 채점 샘플은 11~20개로 적으며, 점수는 aux 모델의 개념 추출과 반사실 생성에 의존하므로 일반적인 모델 우열로 해석하기 어렵습니다. `ALL__excluded_samples.csv`에는 답변 추출 오류나 반사실 생성 오류·경고 등의 사유로 제외된 샘플이 기록되어 있으며, 이들은 표의 n과 평균에 포함되지 않습니다.

![Cache 9 run별 요약 그래프](pubmedqa_wtt_batch_cache%209/ALL__run_summary_plot.png)

`pubmedqa_wtt_batch_cache 9/`의 `ALL__run_summary_plot.png`에서 가져온 요약 그래프입니다.

## 실행

`0521_intro2genai_demo_batched.ipynb`로 실행합니다.
- 로컬 모델은 Ollama로, `google/...`·`gemini...` 모델은 `google.colab.ai`로 자동 라우팅(`llm_generate`가 모델 이름 보고 분기).
- setup(패키지 + Ollama 설치) → 파이프라인 정의(pydantic 스키마 + 본체) → `run_colab_experiment(...)` 실행 → 결과 CSV 확인

전역변수 목록:

- `TARGET_MODELS` — 답변 모델. Ollama와 Colab 제공 gemini 교차 사용 가능
- `AUX_MODELS` — concept 추출 + counterfactual 생성 담당하는 보조 LLM(기본 `gemma4:e4b`).
- 둘의 모든 조합마다 run 하나(`target_{...}__aux_{...}`)가 되어 순차 실행됨
- `N_SAMPLES`, `MAX_CONCEPTS_PER_SAMPLE`, `OVERWRITE_CACHE` 등은 `run_colab_experiment()` 인자로 조절.

Colab 디스크가 빠듯해서, 파이프라인을 4단계(original → concept/cf 생성 → cf answer → 집계)로 나눠 모델별 배치 처리합니다. 
안 쓰는 target 모델은 디스크 압박 시 자동 삭제하고 aux는 보호합니다.

데이터는 HuggingFace `qiaojin/PubMedQA`의 `pqa_labeled`를 쓰고, context 길이와 gold label 유무로 필터링합니다.

## 결과물

각 run마다 캐시 폴더에 실험결과 저장됩니다.

| 파일 | 내용 |
|---|---|
| `*__original_answers.jsonl` | 원본 context에 대한 답변 |
| `*__aux_counterfactuals.jsonl` | 추출된 concept + counterfactual context |
| `*__counterfactual_answers.jsonl` | 바뀐 context에 대한 답변 |
| `*__concept_table.csv` | 위 셋을 concept 단위로 조인 (답 바뀌었는지 포함) |
| `*__sample_summary.csv` | 샘플 단위 집계 + `walk_the_talk_lite_score` |
| `*__excluded_samples.csv` | 채점 제외된 샘플과 사유 (cache 9부터) |
| `ALL__*` | 폴더 내 전체 run 합친 집계 + 요약 그래프 |

최신 결과는 `pubmedqa_wtt_batch_cache 9/`. cache 8보다 target 조합이 많고 `excluded_samples`가 추가되어 있습니다.

## 레거시

루트의 `combined_cache_7_8_9__*.csv`, `LLM_Faithfulness_Comparison*.png`, 낱개로 남은 `target_*__aux_*__*.jsonl`은 예전 분석 패스 산출물이고, 만들었던 노트북들은 정리하면서 지웠으며 결과만 참고용으로 남겨두었습니다. 재현하려면 메인 노트북을 다시 돌려 새 캐시를 만들면 됩니다.
