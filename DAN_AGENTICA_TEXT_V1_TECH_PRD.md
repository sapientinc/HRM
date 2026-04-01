# Dan Agentica: Text V1 Technical Product Requirements Document

**Master Agent for Text Data Pipeline**  
*Generation → Cleaning → Annotation → Segmentation → Ready-to-Train*

---

## EXECUTIVE SUMMARY

**Product Name**: Dan Agentica (Text Edition)

**Vision**: One command-line tool + Web dashboard that transforms raw task descriptions into production-ready NLP datasets in minutes.

**Core Problem Solved**:
- Creating labeled text datasets costs $10K-$50K via Scale AI or Labelbox
- Manual annotation takes 2-4 months for 10K samples
- Quality is inconsistent (inter-annotator agreement: 0.65-0.80)
- No feedback loops between generation and annotation

**Our Solution**:
- **Input**: Task description (e.g., "Medical NER dataset: drugs, procedures, diagnoses")
- **Output**: 10K fully labeled, validated, ready-to-train dataset in 2-4 hours
- **Cost**: $50 (vs $10,000)
- **Quality**: 0.88+ F1-score (near-human)

**Key Differentiator**: Master Agent orchestrates 5 stages with intelligent feedback loops, retry logic, and self-improvement.

---

## PART 1: RESEARCH FINDINGS

### 1.1 State of Synthetic Text Generation (2025)

**What Works Well** ✅:

1. **LLM-Based Generation** [web:59][web:67]
   - GPT-4, Claude 3.5 can generate task-relevant text
   - Cost: ~$0.002 per sample (API)
   - Quality: 85-95% for objective tasks (topic classification, NER)
   - Quality: 60-75% for subjective tasks (sentiment, humor) [web:78]

2. **Entity-Based NER Generation** [web:71][web:75]
   - Provide entity list → LLM generates context sentences
   - F1-score: 0.80-0.88 on biomedical NER [web:75][web:84]
   - Much better than free-form generation

3. **Self-Refinement via Reflection** [web:8][web:62]
   - Multi-agent reflection improves quality 15-25%
   - GRAID approach: constraints + reflective augmentation [web:8]
   - ARISE: rule induction + data generation loop [web:62]

4. **Differential Privacy for Sensitive Data** [web:59][web:61]
   - DP-KPS: generate synthetic healthcare text without privacy leakage [web:59]
   - Balance framework: fairness-aware synthesis [web:61]

**What Doesn't Work** ❌:

1. **Hallucination & Factual Errors** [web:67]
   - LLMs make up facts, miss edge cases
   - Requires aggressive filtering + validation

2. **Low Task Diversity** [web:67]
   - Single prompt → all samples look similar
   - Need: prompt variation, negative examples, edge cases

3. **Subjective Tasks Fail** [web:78]
   - Humor, sarcasm, sentiment: <70% quality
   - Objective tasks: 90%+ quality

### 1.2 Text Annotation Benchmarks

**NER Task Performance** [web:70][web:75][web:84]:
- Human agreement (inter-annotator): 0.78-0.92 F1
- GPT-4 + prompting: 0.84-0.91 F1 [web:69]
- GPT-4-Vision (multimodal): Higher accuracy on OCR tasks
- Local fine-tuned models: 0.80-0.86 F1

**Time & Cost** [web:67]:
- Manual annotation: $0.50-$2.00 per sample
- Scale AI: $0.50-$1.50 per sample
- LLM-based: $0.001-$0.01 per sample

**Quality by Task Type**:
- Classification (objective): 90%+ accuracy
- NER (entity extraction): 88%+ F1
- Relation extraction: 80%+ F1
- Sentiment (subjective): 65-75% accuracy

### 1.3 Data Cleaning Challenges

**Top Issues** [web:77][web:83]:
1. Text normalization (case, punctuation, contractions)
2. Missing values & incomplete sentences
3. Encoding errors (UTF-8, special chars)
4. Duplicate detection
5. Outlier/noise removal (hallucinations, off-topic)

**Validation Framework** [web:77]:
- Schema-based validation (Pydantic)
- Custom business rules
- Statistical bounds (length, token count)
- Pattern matching (regex, fuzzy matching)

---

## PART 2: SYSTEM ARCHITECTURE

### 2.1 Master Agent High-Level Overview

```mermaid
graph TB
    User["👤 User Input<br/>Task Description"]
    
    User -->|"Create dataset"| Dashboard["🖥️ Dashboard / CLI"]
    
    Dashboard -->|"Task JSON"| MasterAgent["🧠 MASTER AGENT<br/>Orchestrator + Learner"]
    
    MasterAgent -->|"dispatch"| Queue["📋 Task Queue<br/>Redis + Celery"]
    
    Queue -->|"Stage 1"| Gen["🔄 GENERATION<br/>LLM Synthesis"]
    Queue -->|"Stage 2"| Clean["🔄 CLEANING<br/>Validation + Dedup"]
    Queue -->|"Stage 3"| Anno["🔄 ANNOTATION<br/>NER/Classification"]
    Queue -->|"Stage 4"| Seg["🔄 SEGMENTATION<br/>Entity Tagging"]
    Queue -->|"Stage 5"| QA["🔄 QA<br/>Consensus Check"]
    
    Gen -->|"10K texts"| S3_Gen["S3 Gen Bucket"]
    Clean -->|"9.5K valid"| S3_Clean["S3 Clean Bucket"]
    Anno -->|"9.5K annotated"| S3_Anno["S3 Anno Bucket"]
    Seg -->|"9.5K tagged"| S3_Seg["S3 Seg Bucket"]
    QA -->|"9K final"| S3_Final["S3 Final Bucket"]
    
    S3_Final -->|"quality report"| MasterAgent
    MasterAgent -->|"feedback"| Gen
    
    S3_Final -->|"download"| User
    
    style MasterAgent fill:#e3f2fd
    style Gen fill:#fff3e0
    style Clean fill:#f3e5f5
    style Anno fill:#e8f5e9
    style Seg fill:#fce4ec
    style QA fill:#e0f2f1
```

### 2.2 Master Agent Decision Engine

```mermaid
graph TD
    Start["🎯 Task Submitted<br/>task_id: gen_xyz<br/>type: 'medical_ner'<br/>count: 10000"] 
    
    Start --> Init["Initialize State<br/>status: QUEUED<br/>completed: 0<br/>quality_score: 0.0"]
    
    Init --> Check1{"Stage 1<br/>Generation<br/>Complete?"}
    
    Check1 -->|"No"| Dispatch1["Dispatch Generation<br/>Prompt engineering<br/>LLM API calls<br/>10 variations"]
    
    Dispatch1 --> Monitor1["Monitor Generation<br/>Track: throughput<br/>cost, errors<br/>WebSocket updates"]
    
    Monitor1 --> Complete1{"Generated<br/>Count?"}
    
    Complete1 -->|"< 10000"| Monitor1
    Complete1 -->|">= 10000"| Decide1
    
    Check1 -->|"Yes"| Decide1{"Quality Gate 1<br/>Pass Rate?<br/>✓ >= 95%"}
    
    Decide1 -->|"✗ < 95%"| Analyze1["Analyze Failures<br/>- Prompt issue?<br/>- LLM hallucination?<br/>- Task ambiguous?"]
    
    Analyze1 --> Retry1{"Try<br/>Regenerate?"}
    
    Retry1 -->|"Yes"| Adapt1["Adapt Prompts<br/>- More examples<br/>- Better instructions<br/>- Add constraints"]
    
    Adapt1 --> Monitor1
    
    Retry1 -->|"No"| Continue1["Accept & Continue<br/>with warnings"]
    
    Decide1 -->|"✓ >= 95%"| Check2
    Continue1 --> Check2{"Stage 2<br/>Cleaning<br/>Complete?"}
    
    Check2 -->|"No"| Dispatch2["Dispatch Cleaning<br/>Validation rules<br/>Dedup + outlier removal<br/>Format standardization"]
    
    Dispatch2 --> Monitor2["Monitor Cleaning<br/>Track: pass rate<br/>rejections, reasons"]
    
    Monitor2 --> Complete2{"Cleaned<br/>Count?"}
    
    Complete2 -->|"< 9500"| Monitor2
    Complete2 -->|">= 9500"| Decide2
    
    Decide2{"Quality Gate 2<br/>Validity?<br/>✓ >= 90%"}
    
    Decide2 -->|"✗ < 90%"| Analyze2["Analyze Cleaning<br/>- Too strict rules?<br/>- Bad generation?<br/>- Text length issue?"]
    
    Analyze2 --> Adjust2["Adjust Validation<br/>Rules & Thresholds"]
    
    Adjust2 --> Monitor2
    
    Decide2 -->|"✓ >= 90%"| Check3{"Stage 3<br/>Annotation<br/>Complete?"}
    
    Check3 -->|"No"| Dispatch3["Dispatch Annotation<br/>GPT-4 with prompts<br/>Few-shot examples<br/>Entity templates"]
    
    Dispatch3 --> Monitor3["Monitor Annotation<br/>Track: annotation rate<br/>token cost"]
    
    Monitor3 --> Complete3{"Annotated<br/>Count?"}
    
    Complete3 -->|"< 9500"| Monitor3
    Complete3 -->|">= 9500"| Check4
    
    Check4{"Stage 4<br/>Segmentation<br/>Complete?"}
    
    Check4 -->|"No"| Dispatch4["Dispatch Segmentation<br/>Parse annotations<br/>Validate entity spans<br/>Resolve conflicts"]
    
    Dispatch4 --> Monitor4["Monitor Segmentation<br/>Track: parsing errors<br/>span validity"]
    
    Monitor4 --> Complete4{"Segmented<br/>Count?"}
    
    Complete4 -->|"< 9500"| Monitor4
    Complete4 -->|">= 9500"| Check5
    
    Check5{"Stage 5<br/>QA Check<br/>Complete?"}
    
    Check5 -->|"No"| Dispatch5["Dispatch QA<br/>Consistency checks<br/>F1-score on validation<br/>Sample-level review"]
    
    Dispatch5 --> Monitor5["Monitor QA<br/>Track: pass rate<br/>common issues"]
    
    Monitor5 --> Complete5{"QA<br/>Pass?"}
    
    Complete5 -->|"< 9000"| Decide5
    Complete5 -->|">= 9000"| Success
    
    Decide5{"Retry<br/>or Accept?"}
    
    Decide5 -->|"Retry best stage"| Analyze5["Learn from Failures<br/>Record metrics<br/>Update models"]
    
    Analyze5 --> Monitor1
    
    Decide5 -->|"Accept"| Success["✅ SUCCESS<br/>9000 ready-to-train<br/>Generate report"]
    
    Success --> Export["Export Dataset<br/>JSONL format<br/>Validation metrics<br/>Audit trail"]
    
    Export --> Deliver["📦 Deliver to User<br/>S3 signed URL<br/>Quality report<br/>Cost breakdown"]
    
    style Start fill:#c8e6c9
    style Success fill:#c8e6c9
    style Deliver fill:#c8e6c9
    style Decide1 fill:#fff9c4
    style Decide2 fill:#fff9c4
    style Decide5 fill:#fff9c4
    style Analyze1 fill:#ffccbc
    style Analyze2 fill:#ffccbc
    style Adapt1 fill:#ffccbc
```

### 2.3 Full Pipeline Flow (5-Stage Detail)

```mermaid
graph LR
    subgraph Stage1["Stage 1: GENERATION"]
        S1A["Task Description<br/>Medical NER"]
        S1B["Prompt Engineer<br/>Expand to 10 variations"]
        S1C["LLM Synthesis<br/>GPT-4 or Claude<br/>Generate texts"]
        S1D["10K Texts<br/>+ metadata<br/>seed, prompt, model"]
        
        S1A --> S1B --> S1C --> S1D
    end
    
    subgraph Stage2["Stage 2: CLEANING"]
        S2A["10K Raw Texts"]
        S2B["Format Validation<br/>- Length check<br/>- Encoding valid<br/>- No corruption"]
        S2C["Deduplication<br/>Fuzzy match<br/>Remove exact dupes"]
        S2D["Outlier Detection<br/>- Too short/long<br/>- Gibberish<br/>- Off-domain"]
        S2E["Text Normalization<br/>- Case standardization<br/>- Punctuation<br/>- Special chars"]
        S2F["9.5K Clean Texts<br/>Quality: 0.95"]
        
        S2A --> S2B --> S2C --> S2D --> S2E --> S2F
    end
    
    subgraph Stage3["Stage 3: ANNOTATION"]
        S3A["9.5K Clean Texts"]
        S3B["LLM Annotation<br/>GPT-4-Vision<br/>with few-shot examples"]
        S3C["Extract Entities<br/>Drug, Procedure<br/>Diagnosis"]
        S3D["Confidence Scoring<br/>High conf: 0.90+<br/>Low conf: re-annotate"]
        S3E["9.5K Annotated<br/>Entities + spans"]
        
        S3A --> S3B --> S3C --> S3D --> S3E
    end
    
    subgraph Stage4["Stage 4: SEGMENTATION"]
        S4A["9.5K Annotated"]
        S4B["Parse Spans<br/>Extract character offsets<br/>BIO tagging"]
        S4C["Validate Spans<br/>- No overlaps<br/>- Valid boundaries<br/>- Type consistency"]
        S4D["Resolve Conflicts<br/>Multiple tags<br/>Choose best"]
        S4E["9.5K Tagged<br/>BIO format<br/>CoNLL format"]
        
        S4A --> S4B --> S4C --> S4D --> S4E
    end
    
    subgraph Stage5["Stage 5: QA"]
        S5A["9.5K Tagged"]
        S5B["Validation Checks<br/>- Span validity<br/>- Entity consistency<br/>- Label distribution"]
        S5C["Sample Review<br/>LLM checks<br/>100 random samples"]
        S5D["F1-Score Calc<br/>vs gold standard<br/>micro + macro"]
        S5E["9K Final<br/>F1-score: 0.88<br/>Ready to train"]
        
        S5A --> S5B --> S5C --> S5D --> S5E
    end
    
    Stage1 --> Stage2
    Stage2 --> Stage3
    Stage3 --> Stage4
    Stage4 --> Stage5
    
    style Stage1 fill:#fff3e0
    style Stage2 fill:#f3e5f5
    style Stage3 fill:#e8f5e9
    style Stage4 fill:#fce4ec
    style Stage5 fill:#e0f2f1
```

### 2.4 Stage 1: Generation (Detailed)

```mermaid
graph TD
    Task["Task Input<br/>Domain: medical<br/>Type: NER<br/>Count: 10000"]
    
    Task --> PromptEng["PROMPT ENGINEERING<br/>LLM: Claude 3.5"]
    
    PromptEng --> Template["Template Generation<br/>Example task prompts"]
    
    Template --> Variation["Generate 10 Variations<br/>- Different angles<br/>- Different styles<br/>- Diverse examples"]
    
    Variation --> Prompts["10 Prompt Templates<br/>1. Standard NER format<br/>2. Dialogue format<br/>3. Case-study format<br/>...10. Edge cases"]
    
    Prompts --> GenLoop["GENERATION LOOP<br/>For each prompt × 1000 samples"]
    
    GenLoop --> APICall["Call LLM API<br/>Model: GPT-4<br/>Temp: 0.8 (diversity)<br/>Max tokens: 200"]
    
    APICall --> Generated["Generated Text<br/>+ seed<br/>+ prompt_id<br/>+ model"]
    
    Generated --> PostProc["POST-PROCESSING<br/>- Extract text<br/>- Log metadata<br/>- Save to S3"]
    
    PostProc --> Output["OUTPUT: 10K Texts<br/>metadata.jsonl"]
    
    Output --> Cost["Cost Tracking<br/>$0.002/sample<br/>Total: $20"]
    
    style PromptEng fill:#fff3e0
    style Variation fill:#fff3e0
    style GenLoop fill:#fff3e0
    style APICall fill:#ffe082
    style Output fill:#c8e6c9
    style Cost fill:#ffccbc
```

### 2.5 Stage 2: Cleaning (Detailed)

```mermaid
graph TD
    Input["10K Raw Texts<br/>From Generation"]
    
    Input --> Step1["STEP 1: Format Validation<br/>Check encoding<br/>No null bytes<br/>Valid UTF-8"]
    
    Step1 --> ValidPass{"UTF-8<br/>Valid?"}
    
    ValidPass -->|"✓ Yes"| Step2
    ValidPass -->|"✗ No"| Reject1["REJECT<br/>Log: encoding_error"]
    
    Step2["STEP 2: Length Check<br/>Min: 10 tokens<br/>Max: 500 tokens"]
    
    Step2 --> LenPass{"Length<br/>OK?"}
    
    LenPass -->|"✓ Yes"| Step3
    LenPass -->|"✗ No"| Reject2["REJECT<br/>Log: length_error"]
    
    Step3["STEP 3: Deduplication<br/>Exact match<br/>Fuzzy match (0.95)"]
    
    Step3 --> DedupPass{"Unique<br/>Text?"}
    
    DedupPass -->|"✓ Yes"| Step4
    DedupPass -->|"✗ Duplicate"| Reject3["REJECT<br/>Log: duplicate"]
    
    Step4["STEP 4: Outlier Detection<br/>Gibberish test<br/>Perplexity score<br/>Domain check"]
    
    Step4 --> OutPass{"Valid<br/>Content?"}
    
    OutPass -->|"✓ Yes"| Step5
    OutPass -->|"✗ Outlier"| Reject4["REJECT<br/>Log: outlier"]
    
    Step5["STEP 5: Normalization<br/>Lowercase<br/>Fix spacing<br/>Remove extra punctuation"]
    
    Step5 --> Output["OUTPUT: 9.5K Clean<br/>Pass rate: 95%<br/>Rejected: 500"]
    
    Reject1 --> Stats["Rejection Stats<br/>Encoding: 2%<br/>Length: 1%<br/>Duplicate: 1%<br/>Outlier: 1%"]
    
    Reject2 --> Stats
    Reject3 --> Stats
    Reject4 --> Stats
    
    Stats --> Report["Report to Master Agent<br/>Pass rate: 0.95<br/>Continue to Stage 3"]
    
    style Step1 fill:#f3e5f5
    style Step2 fill:#f3e5f5
    style Step3 fill:#f3e5f5
    style Step4 fill:#f3e5f5
    style Step5 fill:#f3e5f5
    style Output fill:#c8e6c9
    style Stats fill:#ffccbc
```

### 2.6 Stage 3: Annotation (Detailed)

```mermaid
graph TD
    Input["9.5K Clean Texts"]
    
    Input --> Prepare["PREPARE FOR LLM<br/>Create few-shot examples<br/>Drug: ibuprofen, aspirin<br/>Procedure: biopsy, CT scan<br/>Diagnosis: diabetes, pneumonia"]
    
    Prepare --> Prompt["BUILD ANNOTATION PROMPT<br/>System: 'You are medical NER expert'<br/>Few-shot: 3-5 examples<br/>User: text + instruction"]
    
    Prompt --> GenLoop["ANNOTATION LOOP<br/>For each text"]
    
    GenLoop --> LLMCall["Call LLM (GPT-4)<br/>Prompt + few-shots<br/>Max tokens: 500<br/>Temp: 0.0 (deterministic)"]
    
    LLMCall --> Parse["PARSE OUTPUT<br/>Extract entities<br/>Validate format"]
    
    Parse --> ParsePass{"Valid<br/>JSON?"}
    
    ParsePass -->|"✓ Yes"| Extract
    ParsePass -->|"✗ No"| Retry["Retry with<br/>different format"]
    
    Retry --> LLMCall
    
    Extract["EXTRACT ENTITIES<br/>Entity name<br/>Entity type<br/>Confidence score"]
    
    Extract --> Confidence{"Conf<br/>> 0.85?"}
    
    Confidence -->|"✓ High"| Accept["ACCEPT<br/>Store annotation"]
    Confidence -->|"✗ Low"| LowConf["Mark low-confidence<br/>Flag for review"]
    
    Accept --> Output["OUTPUT: 9.5K Annotated<br/>Annotation format:<br/>{<br/>  'text': '...',<br/>  'entities': [<br/>    {<br/>      'text': 'ibuprofen',<br/>      'type': 'DRUG',<br/>      'confidence': 0.92<br/>    }<br/>  ]<br/>}"]
    
    LowConf --> Output
    
    Output --> Cost["Cost Tracking<br/>$0.0005/sample<br/>Total: $5"]
    
    style Prepare fill:#e8f5e9
    style Prompt fill:#e8f5e9
    style GenLoop fill:#e8f5e9
    style LLMCall fill:#a5d6a7
    style Extract fill:#e8f5e9
    style Output fill:#c8e6c9
    style Cost fill:#ffccbc
```

### 2.7 Stage 4: Segmentation (Detailed)

```mermaid
graph TD
    Input["9.5K Annotated"]
    
    Input --> Step1["STEP 1: Parse Annotations<br/>Extract entity spans<br/>Entity text<br/>Entity type"]
    
    Step1 --> Step2["STEP 2: Find Offsets<br/>Search text for<br/>entity occurrence<br/>Start + end indices"]
    
    Step2 --> Found{"Span<br/>Found?"}
    
    Found -->|"✓ Yes"| Step3
    Found -->|"✗ No"| Fuzzy["Fuzzy match<br/>or reject"]
    
    Step3["STEP 3: BIO Tagging<br/>B-DRUG (begin)<br/>I-DRUG (inside)<br/>O (outside)"]
    
    Step3 --> Step4["STEP 4: Validate Spans<br/>- No overlaps<br/>- Valid boundaries<br/>- Consistent typing"]
    
    Step4 --> Valid{"Valid<br/>Span?"}
    
    Valid -->|"✓ Yes"| Step5
    Valid -->|"✗ Conflict"| Resolve["Resolve conflict<br/>Keep highest conf<br/>or merge"]
    
    Step5["STEP 5: Format Output<br/>CoNLL-2003 format<br/>One token per line<br/>Token<br/>Label"]
    
    Step5 --> Example["EXAMPLE OUTPUT:<br/>The O<br/>patient O<br/>took B-DRUG<br/>ibuprofen I-DRUG<br/>for O<br/>pain O"]
    
    Resolve --> Output["OUTPUT: 9.5K Tagged<br/>CoNLL format<br/>JSONL format<br/>Ready for ML training"]
    
    style Step1 fill:#fce4ec
    style Step2 fill:#fce4ec
    style Step3 fill:#fce4ec
    style Step4 fill:#fce4ec
    style Step5 fill:#fce4ec
    style Output fill:#c8e6c9
```

### 2.8 Stage 5: QA (Detailed)

```mermaid
graph TD
    Input["9.5K Tagged Data"]
    
    Input --> Check1["CHECK 1: Format<br/>Valid CoNLL?<br/>No parsing errors<br/>All tokens present"]
    
    Check1 --> Pass1{"Format<br/>OK?"}
    
    Pass1 -->|"✗ No"| Reject1["REJECT<br/>Log format_error"]
    Pass1 -->|"✓ Yes"| Check2
    
    Check2["CHECK 2: Label Validity<br/>Entity types match spec<br/>No invalid combos<br/>I- without B-"]
    
    Check2 --> Pass2{"Labels<br/>Valid?"}
    
    Pass2 -->|"✗ No"| Reject2["REJECT<br/>Log: invalid_label"]
    Pass2 -->|"✓ Yes"| Check3
    
    Check3["CHECK 3: Span Consistency<br/>Entity text matches<br/>Correct type<br/>No partial matches"]
    
    Check3 --> Pass3{"Spans<br/>Consistent?"}
    
    Pass3 -->|"✗ No"| Reject3["REJECT<br/>Log: span_mismatch"]
    Pass3 -->|"✓ Yes"| Check4
    
    Check4["CHECK 4: Sample Review<br/>LLM validates<br/>100 random samples<br/>Manual spot check"]
    
    Check4 --> Pass4{"Good<br/>Quality?"}
    
    Pass4 -->|"✗ No"| Report4["Report issues<br/>Update prompts<br/>Retry generation"]
    Pass4 -->|"✓ Yes"| Check5
    
    Check5["CHECK 5: Metrics Calculation<br/>Precision, Recall, F1<br/>Per-entity stats<br/>Label distribution"]
    
    Check5 --> Metrics["METRICS OUTPUT<br/>Overall F1: 0.88<br/>Per-entity F1:<br/>  DRUG: 0.91<br/>  PROCEDURE: 0.85<br/>  DIAGNOSIS: 0.87<br/>Label dist:<br/>  DRUG: 35%<br/>  PROCEDURE: 40%<br/>  DIAGNOSIS: 25%"]
    
    Metrics --> Output["OUTPUT: 9K Final<br/>Quality score: 0.88<br/>Ready to train<br/>Full audit trail"]
    
    Reject1 --> Stats["REJECTION STATS<br/>Format errors: 0.1%<br/>Label errors: 0.2%<br/>Span issues: 0.2%<br/>Sample QA: 0.0%"]
    Reject2 --> Stats
    Reject3 --> Stats
    
    Stats --> Report["Report to Master Agent<br/>Pass rate: 0.90<br/>Quality OK<br/>Proceed to export"]
    
    Report4 --> Retry["FEEDBACK LOOP<br/>Record failure<br/>Suggest improvements<br/>Ready to retry"]
    
    style Check1 fill:#e0f2f1
    style Check2 fill:#e0f2f1
    style Check3 fill:#e0f2f1
    style Check4 fill:#e0f2f1
    style Check5 fill:#e0f2f1
    style Output fill:#c8e6c9
    style Metrics fill:#fff9c4
```

### 2.9 Data Flow (End-to-End)

```mermaid
graph LR
    User["👤 User<br/>Task: 10K medical NER"]
    
    User -->|"API POST"| API["API Gateway<br/>FastAPI<br/>JWT auth"]
    
    API -->|"validate & store"| RDS["PostgreSQL<br/>task_id<br/>user_id<br/>parameters<br/>status"]
    
    API -->|"enqueue"| Redis["Redis Queue<br/>task_xyz<br/>priority: normal<br/>retry: 3"]
    
    Redis -->|"worker 1"| Gen["Worker Gen<br/>Stage 1<br/>Generate 10K"]
    Redis -->|"worker 2"| Clean["Worker Clean<br/>Stage 2<br/>Validate"]
    Redis -->|"worker 3"| Anno["Worker Anno<br/>Stage 3<br/>Annotate"]
    Redis -->|"worker 4"| Seg["Worker Seg<br/>Stage 4<br/>Tag entities"]
    Redis -->|"worker 5"| QA["Worker QA<br/>Stage 5<br/>Validate"]
    
    Gen -->|"10K texts"| S3Gen["S3 Gen Bucket<br/>stage1/"]
    Clean -->|"reads"| S3Gen
    Clean -->|"9.5K clean"| S3Clean["S3 Clean Bucket<br/>stage2/"]
    
    Anno -->|"reads"| S3Clean
    Anno -->|"9.5K anno"| S3Anno["S3 Anno Bucket<br/>stage3/"]
    
    Seg -->|"reads"| S3Anno
    Seg -->|"9.5K tagged"| S3Seg["S3 Seg Bucket<br/>stage4/"]
    
    QA -->|"reads"| S3Seg
    QA -->|"9K final"| S3Final["S3 Final Bucket<br/>stage5/<br/>dataset.jsonl<br/>metadata.json<br/>report.json"]
    
    Gen -->|"status"| WebSocket["WebSocket<br/>Real-time updates"]
    Clean -->|"progress"| WebSocket
    Anno -->|"progress"| WebSocket
    Seg -->|"progress"| WebSocket
    QA -->|"progress"| WebSocket
    
    WebSocket -->|"broadcast"| Dashboard["Web Dashboard<br/>Status page<br/>Progress bar<br/>Live metrics"]
    
    S3Final -->|"generate URL"| SignedURL["Signed S3 URL<br/>24h expiry"]
    
    SignedURL -->|"send link"| User
    Dashboard -->|"display"| User
    
    RDS -->|"logs"| Monitor["CloudWatch<br/>Logs<br/>Metrics<br/>Alerts"]
    
    style Gen fill:#fff3e0
    style Clean fill:#f3e5f5
    style Anno fill:#e8f5e9
    style Seg fill:#fce4ec
    style QA fill:#e0f2f1
```

### 2.10 State Machine (Complete Task Lifecycle)

```mermaid
stateDiagram-v2
    [*] --> QUEUED: Task created
    
    QUEUED --> GENERATING: Worker assigned
    
    GENERATING --> GEN_DONE: 10K generated
    GENERATING --> GEN_FAIL: API error / timeout
    
    GEN_FAIL --> GEN_RETRY: Retry logic
    GEN_RETRY --> GENERATING: Resubmit
    
    GEN_DONE --> QUALITY_CHECK_1: Evaluate generation
    
    QUALITY_CHECK_1 --> CLEANING: Pass rate > 95%
    QUALITY_CHECK_1 --> ADAPT_GEN: Pass rate < 95%
    
    ADAPT_GEN --> GENERATING: Retrain generation
    
    CLEANING --> CLEAN_DONE: 9.5K cleaned
    CLEANING --> CLEAN_FAIL: Validation error
    
    CLEAN_FAIL --> CLEAN_RETRY: Review rules
    CLEAN_RETRY --> CLEANING: Reprocess
    
    CLEAN_DONE --> QUALITY_CHECK_2: Evaluate cleaning
    
    QUALITY_CHECK_2 --> ANNOTATION: Pass rate > 90%
    QUALITY_CHECK_2 --> ADAPT_CLEAN: Pass rate < 90%
    
    ADAPT_CLEAN --> CLEANING: Adjust rules
    
    ANNOTATION --> ANNO_DONE: 9.5K annotated
    ANNOTATION --> ANNO_FAIL: LLM error
    
    ANNO_FAIL --> ANNO_RETRY: Retry with diff prompt
    ANNO_RETRY --> ANNOTATION: Resubmit
    
    ANNO_DONE --> SEGMENTATION: Parse entities
    
    SEGMENTATION --> SEG_DONE: 9.5K tagged
    SEGMENTATION --> SEG_FAIL: Parse error
    
    SEG_FAIL --> SEG_RETRY: Manual fix
    SEG_RETRY --> SEGMENTATION: Reprocess
    
    SEG_DONE --> QA: Run validation
    
    QA --> QA_PASS: 9K pass QA
    QA --> QA_FAIL: Quality < 0.85
    
    QA_FAIL --> LEARN_FAIL: Record metrics
    
    LEARN_FAIL --> DECIDE_REGEN: Agent analyzes
    
    DECIDE_REGEN --> GENERATING: Regenerate all
    DECIDE_REGEN --> READY: Accept with warnings
    
    QA_PASS --> READY: Dataset ready
    
    READY --> EXPORTING: Generate JSONL
    EXPORTING --> EXPORTED: Signed URL created
    EXPORTED --> [*]: User downloads
    
    GEN_FAIL --> [*]: Aborted after retries
    CLEAN_FAIL --> [*]: Aborted after retries
```

### 2.11 Component Interaction Matrix

```mermaid
graph TB
    subgraph Orchestration["🎮 Orchestration"]
        MA["Master Agent<br/>Planner"]
        TQ["Task Queue<br/>Redis"]
        DB["PostgreSQL<br/>State"]
    end
    
    subgraph Workers["⚙️ Workers"]
        W1["Gen Worker<br/>LLM calls"]
        W2["Clean Worker<br/>Validation"]
        W3["Anno Worker<br/>Annotation"]
        W4["Seg Worker<br/>Tagging"]
        W5["QA Worker<br/>Validation"]
    end
    
    subgraph LLMs["🧠 LLM Services"]
        Claude["Claude 3.5<br/>Generation"]
        GPT4["GPT-4<br/>Annotation"]
        GPT4V["GPT-4-Vision<br/>Sample review"]
    end
    
    subgraph Storage["💾 Storage"]
        S3Gen["S3 Gen"]
        S3Cln["S3 Clean"]
        S3Anno["S3 Anno"]
        S3Seg["S3 Seg"]
        S3Final["S3 Final"]
    end
    
    subgraph Monitoring["📊 Monitoring"]
        WS["WebSocket<br/>Live updates"]
        CW["CloudWatch<br/>Metrics"]
        Dashboard["Dashboard<br/>UI"]
    end
    
    MA -->|"dispatch"| TQ
    TQ -->|"task"| W1
    TQ -->|"task"| W2
    TQ -->|"task"| W3
    TQ -->|"task"| W4
    TQ -->|"task"| W5
    
    W1 -->|"uses"| Claude
    W3 -->|"uses"| GPT4
    W5 -->|"uses"| GPT4V
    
    W1 -->|"write"| S3Gen
    W2 -->|"read/write"| S3Gen
    W2 -->|"write"| S3Cln
    W3 -->|"read/write"| S3Cln
    W3 -->|"write"| S3Anno
    W4 -->|"read/write"| S3Anno
    W4 -->|"write"| S3Seg
    W5 -->|"read/write"| S3Seg
    W5 -->|"write"| S3Final
    
    DB -->|"store state"| MA
    W1 -->|"update status"| DB
    W2 -->|"update status"| DB
    W3 -->|"update status"| DB
    W4 -->|"update status"| DB
    W5 -->|"update status"| DB
    
    W1 -->|"progress"| WS
    W2 -->|"progress"| WS
    W3 -->|"progress"| WS
    W4 -->|"progress"| WS
    W5 -->|"progress"| WS
    
    WS -->|"broadcast"| Dashboard
    DB -->|"metrics"| CW
    
    MA -->|"learn"| DB
    S3Final -->|"feedback"| MA
```

---

## PART 3: API SPECIFICATION

### 3.1 Create Dataset Task

```bash
POST /api/v1/datasets/create

{
    "task_name": "Medical NER Dataset",
    "task_description": "Generate labeled dataset for extracting drugs, procedures, and diagnoses from clinical notes",
    "domain": "medical",
    "task_type": "ner",
    "entities": [
        {
            "name": "DRUG",
            "description": "Pharmaceutical drugs and medications",
            "examples": ["ibuprofen", "aspirin", "metformin"]
        },
        {
            "name": "PROCEDURE", 
            "description": "Medical procedures and tests",
            "examples": ["biopsy", "CT scan", "blood test"]
        },
        {
            "name": "DIAGNOSIS",
            "description": "Medical conditions and diagnoses",
            "examples": ["diabetes", "pneumonia", "hypertension"]
        }
    ],
    "sample_count": 10000,
    "quality_threshold": 0.88,
    "budget": {
        "max_cost_usd": 100,
        "max_time_hours": 4,
        "priority": "quality"  # or "speed"
    },
    "output_format": "jsonl",  # jsonl, conll, csv
    "callback_webhook": "https://your-domain.com/webhook/dataset-complete"
}

RESPONSE 201:
{
    "task_id": "task_abc123xyz",
    "status": "QUEUED",
    "created_at": "2025-12-25T10:30:00Z",
    "estimated_time_minutes": 120,
    "estimated_cost_usd": 45.50,
    "progress_url": "/api/v1/datasets/task_abc123xyz/progress",
    "websocket_url": "wss://platform.danagenti.com/ws/task_abc123xyz"
}
```

### 3.2 Get Progress

```bash
GET /api/v1/datasets/task_abc123xyz/progress

RESPONSE 200:
{
    "task_id": "task_abc123xyz",
    "status": "ANNOTATION",
    "current_stage": 3,
    "overall_progress_percent": 65,
    
    "stages": {
        "generation": {
            "status": "COMPLETE",
            "count": 10000,
            "time_seconds": 1800,
            "cost_usd": 20.00
        },
        "cleaning": {
            "status": "COMPLETE",
            "count": 9500,
            "pass_rate": 0.95,
            "time_seconds": 600,
            "cost_usd": 0.00
        },
        "annotation": {
            "status": "IN_PROGRESS",
            "count_done": 6200,
            "count_total": 9500,
            "progress_percent": 65,
            "estimated_remaining_minutes": 30,
            "cost_so_far_usd": 3.10
        },
        "segmentation": {
            "status": "PENDING",
            "estimated_time_minutes": 20
        },
        "qa": {
            "status": "PENDING",
            "estimated_time_minutes": 15
        }
    },
    
    "quality_metrics": {
        "generation_quality": 0.95,
        "cleaning_pass_rate": 0.95,
        "annotation_confidence": 0.87,
        "overall_score": 0.92
    },
    
    "cost_tracking": {
        "cumulative_cost_usd": 23.10,
        "estimated_total_usd": 45.50,
        "budget_remaining_usd": 54.50
    }
}
```

### 3.3 Download Dataset

```bash
GET /api/v1/datasets/task_abc123xyz/download

RESPONSE 200:
{
    "task_id": "task_abc123xyz",
    "status": "COMPLETE",
    "download_url": "https://s3.amazonaws.com/danagenti-datasets/...",
    "file_size_mb": 42.5,
    "expiry_hours": 24,
    
    "dataset_statistics": {
        "total_samples": 9000,
        "total_entities": 21450,
        "avg_entities_per_sample": 2.38,
        "entity_distribution": {
            "DRUG": {"count": 7875, "percent": 36.7},
            "PROCEDURE": {"count": 8610, "percent": 40.1},
            "DIAGNOSIS": {"count": 4965, "percent": 23.2}
        }
    },
    
    "quality_report": {
        "overall_f1_score": 0.88,
        "per_entity_f1": {
            "DRUG": 0.91,
            "PROCEDURE": 0.85,
            "DIAGNOSIS": 0.87
        },
        "coverage": {
            "precision": 0.89,
            "recall": 0.87
        }
    },
    
    "audit_trail": [
        {
            "stage": "GENERATION",
            "status": "COMPLETE",
            "count": 10000,
            "timestamp": "2025-12-25T10:30:00Z",
            "duration_seconds": 1800,
            "notes": "10 prompt variations, Claude 3.5 API"
        },
        {
            "stage": "CLEANING",
            "status": "COMPLETE",
            "count": 9500,
            "pass_rate": 0.95,
            "timestamp": "2025-12-25T11:00:00Z",
            "duration_seconds": 600,
            "notes": "5 validation rules, 500 rejected"
        },
        ...
    ]
}
```

---

## PART 4: TECHNICAL SPECIFICATIONS

### 4.1 System Requirements

**Development (Local)**:
- Python 3.11+
- 16GB RAM
- GPU optional (NVIDIA for faster inference)
- macOS/Linux/Windows

**Production (AWS)**:
- EC2 instances: 4× t4.2xlarge (16 vCPU, 64GB RAM each)
- RDS: PostgreSQL 15
- ElastiCache: Redis 7.0
- S3: Unlimited storage
- Lambda: For task dispatch
- CloudWatch: Monitoring

**Cost Model**:
- API calls (Claude + GPT-4): $0.002-0.005 per sample
- Infrastructure: $0.001 per sample (amortized)
- **Total per sample: ~$0.007**

### 4.2 Dependencies

```python
# Core
fastapi==0.110+
celery==5.3+
redis==5.0+
sqlalchemy==2.0+
pydantic==2.0+

# ML/NLP
transformers==4.45+
torch==2.1+
openai==1.8+
anthropic==0.28+

# Data
pandas==2.2+
numpy==1.24+
datasets==2.19+

# Utilities
httpx==0.26+
loguru==0.7+
python-dotenv==1.0+

# Testing
pytest==7.4+
pytest-asyncio==0.23+
```

### 4.3 Environment Variables

```bash
# LLM APIs
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# AWS
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
AWS_S3_BUCKET=danagenti-datasets

# Database
DATABASE_URL=postgresql://user:pass@host:5432/danagenti
REDIS_URL=redis://host:6379

# Application
DEBUG=false
LOG_LEVEL=INFO
MAX_WORKERS=4
MAX_RETRIES=3
```

---

## PART 5: PRICING & MONETIZATION

### 5.1 Pricing Tiers

**Free Tier**:
- 500 samples/month
- Medical domain only
- Community support
- 24h dataset retention

**Pro Tier** ($49/month):
- 50K samples/month
- Unlimited domains
- Priority queue (1h wait vs 24h)
- Email support
- 30-day dataset retention
- Custom quality thresholds

**Enterprise** (Custom):
- Unlimited samples
- On-premise deployment
- Dedicated instance
- SLA 99.9%
- Custom integrations
- 90-day retention

**Pay-Per-Sample**:
- $0.01 per additional sample (beyond tier limit)

### 5.2 Sample Cost Breakdown

```
Generation (Claude API):  $0.002/sample
Cleaning (local):          $0.000/sample
Annotation (GPT-4):        $0.0035/sample
Segmentation (local):      $0.000/sample
QA (local + GPT-4-Vision): $0.0005/sample
Infrastructure (amortized): $0.001/sample
───────────────────────────────────
Total per sample:          $0.0075
Margin (40%):              $0.003
Sell price (target):       $0.01
```

### 5.3 Year 1 Revenue Projection

```
Assumption: 10,000 active users by EOY 2025

Tier Distribution:
- Free: 60% (6,000 users) = $0 revenue
- Pro: 35% (3,500 users) × $49/month × 12 = $2,058,000
- Enterprise: 5% (500 users) × $150K/year avg = $75,000,000

Additional:
- Pay-per-sample: 500M samples × $0.001 profit = $500,000

Total Year 1 Revenue: ~$77.5M (conservative: only 10% attainment)
= ~$7.7M (achievable target)
```

---

## PART 6: COMPETITIVE ADVANTAGE

### 6.1 vs. Scale AI / Labelbox

| Feature | Scale AI | Dan Agentica |
|---------|----------|---|
| Cost | $0.50-$2.00/sample | $0.01/sample |
| Speed | 2-4 weeks | 2-4 hours |
| Fully Automated | ❌ No | ✅ Yes |
| Quality | 90-95% | 88%+ |
| Feedback Loops | ❌ No | ✅ Yes |
| Privacy (on-prem) | ❌ No | ✅ Yes |
| Subjective Tasks | ✅ Good | ⚠️ Limited |

### 6.2 vs. LLM APIs (DIY Approach)

**Our Approach**:
- ✅ Fully orchestrated pipeline
- ✅ Quality gates at each stage
- ✅ Automatic retry + improvement
- ✅ Audit trail + transparency
- ✅ One-click export

**DIY Approach**:
- ❌ Manual pipeline building
- ❌ No quality guarantees
- ❌ Time-consuming debugging
- ❌ High engineer cost
- ❌ Requires ML expertise

---

## PART 7: ROADMAP

### Q1 2025 (MVP - Medical NER)
- ✅ Core 5-stage pipeline
- ✅ Web dashboard
- ✅ API v1.0
- ✅ Medical domain only
- ✅ NER task type only
- ✅ 50 beta testers

### Q2 2025
- 🔄 Text Classification support
- 🔄 On-premise deployment
- 🔄 Advanced analytics dashboard
- 🔄 500+ paid users
- 🔄 Custom entity training

### Q3 2025
- 🔲 Relation Extraction
- 🔲 Question-Answering datasets
- 🔲 Multi-language support
- 🔲 Fine-tuning service
- 🔲 2000+ users

### Q4 2025
- 🔲 Multimodal (text + images)
- 🔲 Code generation datasets
- 🔲 Enterprise tier
- 🔲 5000+ users

---

## PART 8: RISK ANALYSIS & MITIGATION

### Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|-----------|
| LLM API downtime | 24h delay | Medium | Fallback to local models |
| Poor generation quality | Low F1-score | Low | Multi-model ensemble |
| Hallucinations | Invalid entities | Medium | Aggressive filtering + validation |
| Privacy concerns | Regulatory issues | Low | On-prem option, DPA |
| Subjective task failure | Low quality | High | Clear guidance + tier limits |
| Cost overruns | Margin pressure | Low | Usage limits + monitoring |

---

## PART 9: SUCCESS METRICS (OKRs)

### Q1 2025
- **OKR 1**: 50 active beta users
- **OKR 2**: Average F1-score 0.85+
- **OKR 3**: API response time <5 seconds
- **OKR 4**: Cost per sample $0.01 (target) vs $0.007 (actual)

### Q2 2025
- **OKR 1**: 500 paid Pro users
- **OKR 2**: NPS score 45+
- **OKR 3**: 95% uptime SLA
- **OKR 4**: 1M samples generated

### Year 1
- **OKR 1**: $5M revenue run rate
- **OKR 2**: 10K active users
- **OKR 3**: 95% customer satisfaction
- **OKR 4**: 500M total samples generated

---

## PART 10: TECHNICAL DEBT & FUTURE WORK

### High Priority
1. **Load testing**: Simulate 1000 concurrent users
2. **Error recovery**: Graceful handling of LLM API failures
3. **Monitoring dashboards**: Real-time system health
4. **Cost optimization**: Reduce per-sample cost to $0.005

### Medium Priority
1. **Local model deployment**: Don't rely on external APIs
2. **Advanced filtering**: ML-based outlier detection
3. **Custom prompts**: User-defined generation instructions
4. **Analytics**: User behavior tracking, performance benchmarks

### Low Priority
1. **Web UI polish**: Better UX/design
2. **Mobile app**: iOS/Android clients
3. **Marketplace**: Pre-built datasets, community models
4. **Academic partnerships**: Research collaborations

---

## CONCLUSION

**Dan Agentica** automates the entire text dataset creation pipeline with:
- **Cost**: 50-100× cheaper than manual labeling
- **Speed**: 2-4 hours vs 2-4 weeks
- **Quality**: 88%+ F1-score with full audit trail
- **Simplicity**: One API call = complete dataset

**MVP Launch**: Q1 2025 (Medical NER)
**Target Users**: 500+ by mid-2025
**Revenue Target**: $5M run rate by EOY 2025

Ready to build? Let's start with the core 5-stage pipeline.

---

**Last Updated**: 2025-12-25  
**Status**: Ready for Development  
**Next Step**: Engineering Sprint Planning
