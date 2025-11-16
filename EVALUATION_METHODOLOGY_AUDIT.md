# Evaluation Methodology Audit Report
## DFORL (Differentiable First-Order Rule Learner) System

**Report Date:** November 2025  
**System Version:** Based on code analysis of the DFORL repository  
**Primary Evaluation Code:** `predict_extract_last_version.py`, function `check_MRR_Hits()`

---

## Executive Summary

This report provides a detailed audit of the evaluation methodology used by the DFORL system for computing link prediction metrics (MRR and Hits@k). The evaluation follows a **filtered ranking protocol** where:

- **Test facts** are explicitly marked with `#TEST` tags in the data files
- **Ranking** is performed by generating all possible candidate entity substitutions
- **Filtering** removes facts that appear anywhere in the dataset (train + test)
- **Ties** are handled by assigning the same rank to predictions with equal probabilities
- **Both head and tail prediction** tasks are evaluated for each test fact

---

## 1. Data Partitioning and Structure

### 1.1 Data File Format

The system uses a unified data file format (`.nl` files) where:
- Each line represents a fact: `relation(entity1, entity2).`
- Test facts are marked with `#TEST` suffix: `relation(entity1, entity2).#TEST`
- Probabilistic facts may include probability annotations (for some datasets)
- No negative examples are stored or used in evaluation

**Example from UMLS dataset:**
```
location_of(acquired_abnormality,experimental_model_of_disease).
isa(alga,entity).
...
interacts_with(steroid,eicosanoid).#TEST
isa(clinical_attribute,conceptual_entity).#TEST
```

### 1.2 Dataset Statistics

Based on analysis of included datasets:

| Dataset | Total Facts | Test Facts | Train Facts | Test Percentage |
|---------|-------------|------------|-------------|-----------------|
| UMLS    | 6,529       | 661        | 5,868       | ~10.1%          |
| Nations | 1,992       | 201        | 1,791       | ~10.1%          |

### 1.3 Data Split Creation

**Implementation Details:**
- Data splits are **pre-created** and stored in the `.nl` files with `#TEST` markers
- The `#TEST` marker is used throughout the codebase to identify test facts
- During training: facts with `#TEST` are excluded from the database (line 261-263 in `create_database()`)
- During testing: facts with `#TEST` are included in the database for evaluation

**Code Reference (predict_extract_last_version.py:261-263):**
```python
if test_mode == False and 'TEST' in single_tri:
    single_tri = f.readline()
    continue
```

### 1.4 Test Set Extraction

The test set is extracted by the `make_relation_entities()` function which:
1. Reads a separate `test.nl` file (line 904)
2. Parses each fact into relation and entity pairs
3. Groups test facts by relation type
4. Returns a dictionary: `{relation_name: [(entity1, entity2), ...]}`

**Code Reference (predict_extract_last_version.py:904):**
```python
test_file_path = data_path + 'test.nl'
test_r2e,_,_ = make_relation_entities(test_file_path)
```

---

## 2. Background Knowledge Available at Test Time

### 2.1 Database Construction

During evaluation, the system constructs a knowledge base using **pyDatalog** (a Datalog implementation):

**What is included:**
1. **All training facts** (facts without `#TEST` marker)
2. **All test facts** (when `test_mode=True`)
3. **All relations** across the dataset
4. **All entities** that appear in the dataset

**What is excluded:**
1. Negative examples (any fact with `-` prefix)
2. Test facts during training phase

**Code Reference (predict_extract_last_version.py:250-273):**
```python
with open(all_trituple_path, 'r') as f:
    single_tri = f.readline()
    while single_tri:
        if '-' in single_tri:  # Do not build negative examples
            single_tri = f.readline()
            continue
        if test_mode == False and 'TEST' in single_tri:
            single_tri = f.readline()
            continue
        # Add fact to database
        + locals()[relation_name](first_entity,second_entity)
```

### 2.2 Entity Universe

The complete set of entities is extracted from the **entire dataset** (both train and test):
- Function: `make_relation_entities()` (line 827-867)
- Extracts all unique entities from `task_name.nl` file (line 910)
- These entities form the candidate space for generating corrupted triples

**Code Reference (predict_extract_last_version.py:910):**
```python
_,all_symbolic_predicate,all_ent = make_relation_entities(data_path +task_name+'.nl')
```

---

## 3. Test Facts Evaluated

### 3.1 Test Fact Selection

**All test facts** marked with `#TEST` are evaluated. There is no sampling or selection applied during the evaluation phase.

### 3.2 Evaluation Mode

The system evaluates **both head and tail prediction** for each test fact:
- For each test triple `(h, r, t)`:
  - **Tail prediction:** Predict `t` given `(h, r, ?)`
  - **Head prediction:** Predict `h` given `(?, r, t)`

**Code Reference (predict_extract_last_version.py:923-929):**
```python
for kept_entity_index in range(2):  # Loop twice: head and tail
    change_entity_index = (kept_entity_index+1) % 2
    kept_entity = pairs[kept_entity_index]
    if change_entity_index == 1:
        corrupted_pairs = list(itertools.product([kept_entity],all_ent))
    else:
        corrupted_pairs = list(itertools.product(all_ent,[kept_entity]))
```

This means:
- Total evaluations = 2 × number of test facts
- Each fact contributes to two ranking tasks

---

## 4. Candidate Generation and Ranking

### 4.1 Candidate Generation Process

For each test fact `(h, r, t)`:

**Tail Prediction (predict t given h, r):**
1. Keep head entity `h` fixed
2. Generate all possible candidates: `(h, r, e)` for all entities `e` in the entity universe
3. Score each candidate using learned logic rules

**Head Prediction (predict h given r, t):**
1. Keep tail entity `t` fixed
2. Generate all possible candidates: `(e, r, t)` for all entities `e` in the entity universe
3. Score each candidate using learned logic rules

**Code Reference (predict_extract_last_version.py:926-929):**
```python
if change_entity_index == 1:
    corrupted_pairs = list(itertools.product([kept_entity],all_ent))
else:
    corrupted_pairs = list(itertools.product(all_ent,[kept_entity]))
```

### 4.2 Scoring Mechanism

Each candidate triple is scored based on **learned logic rules**:

**Process:**
1. Load learned logic programs from file (e.g., `best.pl`)
2. For each rule, extract:
   - Rule body (conjunction of predicates)
   - Rule probability/confidence weight
3. Query the Datalog database to find all substitutions satisfying the rule body
4. If a candidate triple can be derived by any rule, assign it the highest probability from applicable rules

**Code Reference (predict_extract_last_version.py:934):**
```python
symbolic_predicate_prob = check_accuracy_of_logic_program(
    task_name, data_path, relation, result_path, test_file+'.pl',
    all_relation, hit_flag=True, t_arity=2, test_mode=True, 
    hit_test_predicate=symbolic_predicate_value, sample_walk=sample_walk, 
    no_training_mode=True
)
```

**Probability Assignment (predict_extract_last_version.py:591-595):**
```python
if hit_flag == True:
    current_precision_value = target_dic[predicate]
    # if the new precision are larger than the current one, update the value
    if precision_rule[search_index] >= current_precision_value: 
        target_dic[predicate] = precision_rule[search_index]
```

**Key Points:**
- Each candidate receives a probability score (default: 0 if not derivable)
- Multiple rules can fire for the same candidate; the **maximum probability** is used
- Probabilities come from rule confidence values extracted during training

---

## 5. Filtering Methodology

### 5.1 Filtered Ranking Protocol

The system implements a **filtered ranking protocol** to avoid penalizing correct predictions:

**Implementation (predict_extract_last_version.py:869-881):**
```python
def indicator_build(relation, corrupted_pairs, all_symbolic_predicate):
    '''
    The corrupted_pairs has no common elements with the facts in 
    both training and testing datasets
    '''
    symbolic_p = {}
    for i in corrupted_pairs:
        single = relation + '('+i[0]+','+i[1] + ')'
        if single in all_symbolic_predicate:
            continue  # Skip this candidate - it's a known fact
        symbolic_p[single] = 0 
    return symbolic_p
```

### 5.2 What Gets Filtered

**Filtered out (not ranked):**
- Any candidate triple that appears in the **complete dataset** (train + test)
- This includes:
  - All training facts
  - All test facts (including other test facts beyond the one being evaluated)

**Retained for ranking:**
- The target test fact (the correct answer)
- All candidate triples that do not appear anywhere in the dataset

### 5.3 Filter Source

The filter is built from:
- **Source:** The complete `.nl` file containing all facts (line 910)
- **Stored as:** A set called `all_symbolic_predicate_set` (line 913)

**Code Reference (predict_extract_last_version.py:910-913):**
```python
_,all_symbolic_predicate,all_ent = make_relation_entities(data_path +task_name+'.nl')
all_symbolic_predicate_set = set(all_symbolic_predicate)
```

### 5.4 Filtering Behavior

This is a **strict filtered setting** where:
- Known facts from any split (train or test) are removed from ranking
- Only the target triple and unseen triples are ranked
- This is consistent with standard knowledge graph completion evaluation protocols

---

## 6. Rank Generation Methodology

### 6.1 Sorting Process

After filtering and scoring, candidates are sorted by their probability scores:

**Code Reference (predict_extract_last_version.py:937-940):**
```python
list_symbolic = []
for i in symbolic_predicate_prob:
    list_symbolic.append((i,symbolic_predicate_prob[i]))
list_symbolic.sort(key=lambda tup: tup[1],reverse=True)  # Sort descending
```

### 6.2 Rank Assignment Algorithm

The system assigns ranks with special handling for ties and zero probabilities:

**Implementation (predict_extract_last_version.py:941-960):**
```python
if list_symbolic[0][1] == 0:
    ini_rank = 1e8  # Very large rank if top prediction has 0 probability
else:
    ini_rank = 1

symbolic_rank = {}
for i in list_symbolic:
    symbolic_rank[i[0]] = [i[1]]  # Store probability

index_sym = list_symbolic[0][0]
symbolic_rank[index_sym].append(ini_rank)  # First element gets initial rank

index = 0
while index < len(list_symbolic)-1:
    index_sym = list_symbolic[index+1][0]
    if list_symbolic[index+1][1] == 0:
        symbolic_rank[index_sym].append(1e8)  # Zero probability → very large rank
    elif list_symbolic[index][1] <= list_symbolic[index+1][1]:
        symbolic_rank[index_sym].append(ini_rank)  # Same rank if same probability
    else:
        ini_rank += 1  # Increment rank
        symbolic_rank[index_sym].append(ini_rank)
    index += 1
```

**Key Properties:**
1. **Descending order:** Higher probabilities get better (lower) ranks
2. **Rank 1** is the best rank
3. **Ties:** Candidates with identical probabilities receive the same rank
4. **Zero probabilities:** Assigned a very large rank (1e8 ≈ 100,000,000)
5. **Rank progression:** After a tie, the next different score gets the next sequential rank (no rank skipping)

### 6.3 Example Rank Assignment

Consider candidates with probabilities: [0.9, 0.7, 0.7, 0.5, 0.0, 0.0]

**Ranks assigned:**
- Probability 0.9 → Rank 1
- Probability 0.7 → Rank 2 (both candidates)
- Probability 0.5 → Rank 3
- Probability 0.0 → Rank 1e8 (both candidates)

**Note:** This is an **optimistic ranking** where ties receive the better rank.

---

## 7. Tie Handling

### 7.1 Tie Detection

Ties occur when multiple candidates have identical probability scores:

**Code Reference (predict_extract_last_version.py:955-956):**
```python
elif list_symbolic[index][1] <= list_symbolic[index+1][1]:
    symbolic_rank[index_sym].append(ini_rank)  # Same rank
```

### 7.2 Tie Resolution Strategy

**Strategy:** **Optimistic ranking** (assign the best possible rank)

When multiple candidates share the same probability:
- All tied candidates receive the **same rank**
- The rank assigned is the **better (lower) rank**
- After the tie group, ranking continues sequentially without skipping

**Implications:**
- If the target fact ties with others, it gets the benefit of the best rank in that group
- This is the **filtered setting with optimistic tie handling**

### 7.3 Special Case: Zero Probabilities

Candidates with zero probability (not derivable by any rule):
- Assigned rank: `1e8` (100,000,000)
- Treated as "not ranked" or "very poor ranking"
- This effectively removes them from meaningful evaluation

**Rationale:** 
- Zero probability means the logic rules cannot derive the fact
- These should not be considered as valid predictions
- Assigning very large rank ensures they don't contribute positively to metrics

---

## 8. MRR (Mean Reciprocal Rank) Calculation

### 8.1 MRR Formula

For each test fact, the reciprocal rank is computed as:

**MRR = 1 / rank_of_correct_fact**

**Code Reference (predict_extract_last_version.py:961-962):**
```python
corrext_rank = symbolic_rank[correct_predicate][1]
MRR_mean.append(1/corrext_rank)
```

### 8.2 Aggregation

The final MRR is the **arithmetic mean** of all individual reciprocal ranks:

**Code Reference (predict_extract_last_version.py:975):**
```python
MRR_mean_value = sum(MRR_mean)/ len(MRR_mean)
```

**Formula:**
```
MRR = (1/N) × Σ(1/rank_i)
```
where N = total number of predictions (2 × number of test facts)

### 8.3 Example Calculation

Consider 3 test facts with ranks: 1, 2, 5

**Reciprocal ranks:** 1/1 = 1.0, 1/2 = 0.5, 1/5 = 0.2

**MRR:** (1.0 + 0.5 + 0.2) / 3 = 0.567

### 8.4 Edge Cases

**Zero probability predictions:**
- If the correct fact receives rank 1e8 (cannot be derived)
- Reciprocal rank = 1/100000000 ≈ 0.00000001
- Effectively contributes ~0 to MRR

---

## 9. Hits@k Calculation

### 9.1 Hits@k Definition

Hits@k measures the proportion of test facts where the correct answer appears in the **top k** ranked predictions.

**Evaluated k values:** 1, 3, 10 (line 886)

**Code Reference (predict_extract_last_version.py:886):**
```python
hits_number = [1,3,10]
```

### 9.2 Computation Method

For each test fact:
1. Get the rank of the correct answer
2. Check if rank ≤ k
3. If yes, count as a "hit" for Hits@k

**Code Reference (predict_extract_last_version.py:964-968):**
```python
hit_index = 0
for i in hits_number:
    if corrext_rank <= i:
        hits_info[hit_index].append(correct_predicate)
    hit_index += 1
```

### 9.3 Aggregation

**Hits@k = (Number of hits at rank ≤ k) / (Total number of predictions)**

**Code Reference (predict_extract_last_version.py:976-979):**
```python
hit_value = []
for i in hits_info:
    a = len(i)
    hit_value.append(a/total_test)
```

### 9.4 Example Calculation

Consider 10 test facts (20 predictions with head+tail):
- 8 predictions have correct answer in top-1
- 12 predictions have correct answer in top-3  
- 15 predictions have correct answer in top-10

**Metrics:**
- Hits@1 = 8/20 = 0.40
- Hits@3 = 12/20 = 0.60
- Hits@10 = 15/20 = 0.75

### 9.5 Tie Handling in Hits@k

When ties occur at the boundary:
- If correct fact is tied at rank k, it **counts as a hit**
- Because ties receive the optimistic (better) rank
- Example: If 3 facts tie at rank 10, all count for Hits@10

---

## 10. Evaluation Process Flow

### 10.1 Complete Workflow

```
1. Load test facts from test.nl file
   ↓
2. Extract all entities from complete dataset
   ↓
3. For each test fact (h, r, t):
   ↓
   4. For each direction (head and tail prediction):
      ↓
      5. Generate all candidate triples (keep one entity, vary the other)
      ↓
      6. Filter: Remove candidates that appear in train or test set
      ↓
      7. Keep target triple and unseen candidates
      ↓
      8. Score candidates using learned logic rules:
         - Query Datalog database
         - Apply each rule
         - Assign maximum probability if derivable
      ↓
      9. Sort candidates by probability (descending)
      ↓
      10. Assign ranks:
          - Handle ties (same probability = same rank)
          - Zero probability → rank 1e8
      ↓
      11. Extract rank of correct answer
      ↓
      12. Update metrics:
          - Add 1/rank to MRR accumulator
          - Check if rank ≤ k for each k in [1,3,10]
          - Increment hit counters
   ↓
13. Aggregate results:
    - MRR = mean of reciprocal ranks
    - Hits@k = proportion of hits
```

### 10.2 Per-Relation Evaluation

The system evaluates **each relation type separately**:

**Code Reference (predict_extract_last_version.py:915-917):**
```python
for relation in test_r2e:
    logging.info("[MRR HITS]Check relation on:")
    logging.info(relation)
```

**Implications:**
- Different relations may have different test set sizes
- Metrics are computed globally across all relations
- Per-relation metrics could be extracted but are aggregated in the final output

---

## 11. Output and Reporting

### 11.1 Saved Outputs

The evaluation saves multiple files:

**1. MRR Results:**
- File: `NLP/{task_name}/result/MRR{test_file}.dt` (binary pickle)
- File: `NLP/{task_name}/result/MRR{test_file}.txt` (human-readable)
- Contents: List of individual reciprocal ranks and mean MRR

**2. Hits Results:**
- File: `NLP/{task_name}/result/HINT{test_file}.dt` (binary pickle)
- File: `NLP/{task_name}/result/HINT{test_file}.txt` (human-readable)
- Contents: Lists of correct predicates for each k, and Hits@k values

**Code Reference (predict_extract_last_version.py:969-991)**

### 11.2 Output Format

**MRR Text File (`MRR{test_file}.txt`):**
```
2025-11-16 22:17:58.927000
[0.5, 0.333, 1.0, 0.25, ...]  # Individual reciprocal ranks
0.567                          # Mean MRR
instance have checked: 100     # Number of test facts (not predictions)
```

**Hits Text File (`HINT{test_file}.txt`):**
```
2025-11-16 22:17:58.927000
[['rel(e1,e2)', ...], [...], [...]]  # Lists of hits for k=1,3,10
[0.40, 0.60, 0.75]                   # Hits@1, Hits@3, Hits@10
instance have checked: 100           # Number of test facts
```

### 11.3 Logging

Console/log output includes:
- Relation being evaluated
- Test fact being processed
- Final MRR value
- Final Hits@k values
- Success confirmation

**Code Reference (predict_extract_last_version.py:1012-1016):**
```python
logging.info('Check MRR and HINTS for %s success'%task_name)
logging.info('MRR')
logging.info(MRR_mean_value)
logging.info('HITNS@[1,3,10]')
logging.info(hit_value)
```

---

## 12. Comparison with Standard Protocols

### 12.1 Alignment with Knowledge Graph Completion

The DFORL evaluation methodology aligns with **standard filtered ranking protocols** used in knowledge graph completion literature:

**Standard Protocol Elements:**
✅ **Filtered setting:** Remove known facts from ranking  
✅ **Head and tail prediction:** Evaluate both directions  
✅ **MRR metric:** Mean reciprocal rank  
✅ **Hits@k metric:** Top-k accuracy for k ∈ {1, 3, 10}  
✅ **Per-triple evaluation:** Each test fact evaluated independently

### 12.2 Key Characteristics

**Consistent with common practices:**
1. **Filtering includes all known facts** (train + test)
2. **No negative sampling** during evaluation
3. **Complete entity set** used for candidate generation
4. **Optimistic tie handling** (common approach)

**Potential differences from some methods:**
1. **Logic rule-based scoring:** Instead of embedding similarity
2. **Discrete probability scores:** May lead to more ties than continuous embeddings
3. **Zero-probability handling:** Explicit large rank assignment
4. **Per-relation processing:** Sequential evaluation by relation type

### 12.3 Evaluation Setting Classification

Based on the analysis, this evaluation implements:
- **Setting:** Filtered ranking (removes known facts)
- **Tie handling:** Optimistic (best rank in tie group)
- **Coverage:** Both head and tail prediction
- **Scope:** All test facts evaluated

This is equivalent to the **"filtered setting"** used in papers like TransE, DistMult, ComplEx, ConvE, etc.

---

## 13. Important Implementation Details

### 13.1 Test Fact Counting

**Important Note:** The system counts test **facts** but evaluates test **predictions** (which is 2× facts):

**Code Reference (predict_extract_last_version.py:984, 990):**
```python
print('instance have checked:',total_test/2, file=f)
```

- `total_test` increments once per prediction (line 963)
- Output divides by 2 to report number of facts
- Metrics are computed over predictions (not facts)

### 13.2 Database State During Evaluation

**Critical:** When evaluating, the database contains:
- All training facts
- All test facts (included when `test_mode=True`)

This means:
- Logic rules can use test facts in their derivations
- Test facts act as background knowledge during evaluation
- This could be considered **transductive** evaluation

**Code Reference (predict_extract_last_version.py:261-263):**
```python
if test_mode == False and 'TEST' in single_tri:
    single_tri = f.readline()
    continue  # Exclude test facts only during training
```

### 13.3 Rule Probability Extraction

Logic rules include probability/confidence annotations:

**Format in rule files:**
```
relation(X,Y) :- body_predicate1(X,Z) & body_predicate2(Z,Y). # (0.85, 10, 12)
```

Where the tuple `(0.85, 10, 12)` represents:
- 0.85: Probability/confidence of the rule
- 10: Number of correct predictions
- 12: Number of total groundings

**Code Reference (predict_extract_last_version.py:396-400):**
```python
if '#' in rule:
    latter_part = rule[rule.index('#'):]
    rule = rule[:rule.index('#')]
    probability = float(latter_part[latter_part.index('(')+1:latter_part.index(',')])
    precision_rule.append(probability)
```

---

## 14. Potential Considerations for Comparison

When comparing DFORL results with other methods, consider:

### 14.1 Evaluation Protocol Alignment

**Verify that baseline methods use:**
1. Same filtered setting (remove all known facts)
2. Same test set (facts marked with #TEST)
3. Both head and tail prediction
4. Same entity universe
5. Same MRR and Hits@k definitions

### 14.2 Data Leakage Considerations

**Transductive vs. Inductive:**
- DFORL evaluation is **transductive** (test facts available in database)
- Verify if baselines also use transductive setting
- Inductive methods would exclude test facts entirely

### 14.3 Rule-Based vs. Embedding Scoring

**Scoring differences:**
- **DFORL:** Discrete probabilities from logic rules
- **Embeddings:** Continuous similarity scores

**Implications:**
- DFORL may produce more ties (discrete values)
- Tie handling becomes more important
- Zero-probability cases more common

### 14.4 Entity Universe

**Verify baselines use the same entity set:**
- DFORL uses all entities from complete dataset
- Some methods might use different entity subsets
- This affects the difficulty of the ranking task

---

## 15. Summary of Key Findings

### 15.1 Core Evaluation Properties

| **Aspect** | **Implementation** |
|------------|-------------------|
| **Metric Computation** | MRR = mean reciprocal rank; Hits@k = proportion in top-k |
| **Ranking Protocol** | Filtered (removes train + test facts) |
| **Tie Handling** | Optimistic (tied items get best rank) |
| **Evaluation Scope** | Both head and tail prediction |
| **Test Set** | Facts marked with #TEST (~10% of data) |
| **Entity Universe** | All entities in dataset (train + test) |
| **Background Knowledge** | All train facts + all test facts (transductive) |
| **Filtering Source** | Complete dataset (train + test) |
| **Scoring Method** | Logic rule probabilities (discrete) |
| **Zero Probability** | Rank = 1e8 (effectively unranked) |

### 15.2 Alignment with Standard Practices

✅ **Consistent:** Filtered ranking protocol  
✅ **Consistent:** MRR and Hits@k metrics  
✅ **Consistent:** Bidirectional evaluation  
✅ **Consistent:** Complete entity candidate set  
⚠️ **Note:** Transductive setting (test facts in database)  
⚠️ **Note:** Discrete scoring may increase ties  

### 15.3 Recommendations for Fair Comparison

When comparing DFORL with other methods:

1. **Verify evaluation setting:**
   - Ensure baselines use filtered ranking
   - Confirm transductive vs. inductive setting
   - Check if test facts are in database

2. **Check data partitioning:**
   - Use same train/test split
   - Verify #TEST markers are consistently used
   - Confirm same entity universe

3. **Validate metrics:**
   - Ensure same MRR formula (including tie handling)
   - Confirm Hits@k uses same k values [1, 3, 10]
   - Verify both directions evaluated

4. **Document differences:**
   - Note if scoring mechanisms differ
   - Report any variations in protocol
   - Clarify transductive/inductive setting

---

## 16. References

### 16.1 Code Files Analyzed

- `predict_extract_last_version.py` (primary evaluation code)
- `predict_extract.py` (alternative/older version)
- `main.py` (entry point and argument parsing)
- `data_generator.py` (data preparation)
- `tools/choose_test_atoms.py` (test set creation utilities)

### 16.2 Key Functions

- `check_MRR_Hits()` - Main evaluation function (line 884-1017)
- `check_accuracy_of_logic_program()` - Scoring function (line 363-633)
- `create_database()` - Database construction (line 222-274)
- `indicator_build()` - Filtering function (line 869-881)
- `make_relation_entities()` - Data parsing (line 827-867)

### 16.3 Related Publications

Based on README.md citations:
- IJCAI 2022 paper: "Learning First-Order Rules with Differentiable Logic Program Semantics"
- AIJ 2024 paper: "A differentiable first-order rule learner for inductive logic programming"

---

## Appendix A: Evaluation Algorithm Pseudocode

```
FUNCTION evaluate_link_prediction(task_name, test_file):
    # Initialize
    MRR_scores = []
    hits_count = {1: [], 3: [], 10: []}
    
    # Load data
    test_facts = load_test_facts(test_file)
    all_entities = load_all_entities(task_name + '.nl')
    known_facts = load_all_facts(task_name + '.nl')  # train + test
    learned_rules = load_logic_rules(test_file + '.pl')
    
    # Build knowledge base with all facts
    database = create_datalog_database(known_facts)
    
    # Evaluate each test fact in both directions
    FOR each (head, relation, tail) in test_facts:
        FOR direction in [HEAD_PREDICTION, TAIL_PREDICTION]:
            
            # Generate candidates
            IF direction == HEAD_PREDICTION:
                candidates = [(e, relation, tail) for e in all_entities]
                correct = (head, relation, tail)
            ELSE:
                candidates = [(head, relation, e) for e in all_entities]
                correct = (head, relation, tail)
            
            # Filter out known facts (except correct answer)
            filtered_candidates = []
            FOR candidate in candidates:
                IF candidate not in known_facts OR candidate == correct:
                    filtered_candidates.append(candidate)
            
            # Score candidates using logic rules
            scores = {}
            FOR candidate in filtered_candidates:
                max_prob = 0
                FOR rule in learned_rules:
                    IF rule_body_satisfied(rule, candidate, database):
                        max_prob = max(max_prob, rule.probability)
                scores[candidate] = max_prob
            
            # Sort by score (descending)
            sorted_candidates = sort(filtered_candidates, key=scores, reverse=True)
            
            # Assign ranks with tie handling
            rank = 1
            ranks = {}
            prev_score = None
            FOR i, candidate in enumerate(sorted_candidates):
                IF scores[candidate] == 0:
                    ranks[candidate] = 1e8  # Very large rank
                ELIF scores[candidate] == prev_score:
                    ranks[candidate] = rank  # Same rank for tie
                ELSE:
                    IF i > 0 AND prev_score != 0:
                        rank = i + 1
                    ranks[candidate] = rank
                prev_score = scores[candidate]
            
            # Get rank of correct answer
            correct_rank = ranks[correct]
            
            # Update metrics
            MRR_scores.append(1.0 / correct_rank)
            IF correct_rank <= 1:
                hits_count[1].append(correct)
            IF correct_rank <= 3:
                hits_count[3].append(correct)
            IF correct_rank <= 10:
                hits_count[10].append(correct)
    
    # Compute final metrics
    MRR = mean(MRR_scores)
    Hits_at_1 = len(hits_count[1]) / len(MRR_scores)
    Hits_at_3 = len(hits_count[3]) / len(MRR_scores)
    Hits_at_10 = len(hits_count[10]) / len(MRR_scores)
    
    RETURN MRR, Hits_at_1, Hits_at_3, Hits_at_10
```

---

## Appendix B: Example Walkthrough

### Test Fact
`location_of(cell, bacterium).#TEST`

### Step-by-Step Evaluation (Tail Prediction)

**1. Generate Candidates**
- Fixed: `location_of(cell, ?)`
- Candidates: `location_of(cell, e)` for all entities e in dataset
- Assume 50 entities total

**2. Filter**
- Remove candidates appearing in train/test set
- Keep target: `location_of(cell, bacterium)`
- After filtering: 35 candidates remain (15 were known facts)

**3. Score**
- Apply learned rules to each candidate
- Rule 1: `location_of(X,Y) :- part_of(X,Y)` [prob=0.8]
  - Satisfied by 5 candidates (including target)
- Rule 2: `location_of(X,Y) :- in(X,Y) & type(Y,organism)` [prob=0.6]
  - Satisfied by 3 candidates (not including target)

**Target score:** 0.8 (from Rule 1)

**4. Rank**
- Sort candidates by probability
- Scores: [0.9, 0.8, 0.8, 0.8, 0.7, 0.6, 0.6, 0.6, 0.0, 0.0, ...]
- Ranks: [1, 2, 2, 2, 3, 4, 4, 4, 1e8, 1e8, ...]
- Target has probability 0.8 → **Rank 2** (tied with 2 others)

**5. Metrics**
- Reciprocal Rank: 1/2 = 0.5
- Hits@1: No (rank 2 > 1)
- Hits@3: Yes (rank 2 ≤ 3) ✓
- Hits@10: Yes (rank 2 ≤ 10) ✓

**6. Repeat for Head Prediction**
- Predict `location_of(?, bacterium)`
- Same process with different candidate set

---

**End of Report**
