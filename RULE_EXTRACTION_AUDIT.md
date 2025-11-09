# Core Rule Extraction Logic Audit

## Overview
The DFORL system learns first-order logic rules from relational data using a differentiable neural logic program approach. This document outlines the core logic for extracting rule definitions.

## Key Components

### 1. Neural Model Training (`train.py`)
- Trains a neural network model that learns logical rules
- Uses differentiable logic semantics
- Saves trained model to `deepDFOL/{dataset}/result/{predicate}/model`

### 2. Rule Extraction Pipeline

#### Main Entry Point: `get_best_logic_programs()` (predict_extract.py:896)
Iteratively extracts rules at different thresholds to find the best set:
1. Loads trained model
2. Iterates through threshold values (0 to 1, step 0.05)
3. For each threshold:
   - Extracts rules using `extract()`
   - Checks accuracy using `check_accuracy_of_logic_program()`
   - Tracks correctness metrics
4. Selects best threshold based on accuracy
5. Builds final rule set using `build_best_logic_program()`

#### Rule Extraction: `extract()` (predict_extract.py:112)
Converts neural network weights to symbolic logic rules:
1. Loads relation information from `all_relation_dic.dt`
2. Calls `nID.extract_symbolic_logic_programs()` to convert weights
3. Writes rules to `logic_program.pl`

#### Symbolic Extraction: `extract_symbolic_logic_programs()` (nID.py:338)
Core logic for converting neural weights to Prolog rules:
1. Loads valid predicate indices and relation-variable mappings
2. Extracts weights from neural network layers ('Original_layer', 'AUX_layer')
3. Combines and processes layer weights
4. Maps variables to symbols (X, Y, Z, W, M, N, T)
5. Applies threshold to identify active rules
6. Builds Prolog-style rules: `head(X,Y) :- body1(X,Z), body2(Z,Y)`

### 3. Rule Selection and Filtering

#### Build Best Program: `build_best_logic_program()` (predict_extract.py:772)
Filters and saves high-quality rules:
1. Reads all rules from `logic_program.pl`
2. Filters rules based on correctness threshold
3. Annotates rules with accuracy scores
4. Writes to output file (default: `best.pl`)

Output format:
```prolog
head(X,Y) :- body1(X,Z), body2(Z,Y)#(accuracy_score, ...)
```

### 4. Rule Evaluation

#### Accuracy Check: `check_accuracy_of_logic_program()` (predict_extract.py)
Evaluates rules against target facts:
1. Loads rules from prolog file
2. Uses pyDatalog to evaluate rule bodies
3. Checks if rule conclusions match target facts
4. Returns accuracy metrics for each rule

## Output Files

### Generated Files (in `deepDFOL/{dataset}/result/{predicate}/`)
1. **logic_program.pl** - All extracted rules at current threshold
2. **best.pl** (or custom name via `-output_rules`) - Filtered high-quality rules
3. **acc_pred.txt** - Overall accuracy metrics
4. **model/** - Trained neural network

### Rule Format
Rules are written in Prolog syntax:
```prolog
% Standard rule
father(X,Y) :- parent(X,Y), male(X)

% With accuracy annotation
father(X,Y) :- parent(X,Y), male(X)#(0.95, ...)
```

## Command-Line Control

### New Flag: `-output_rules` (default: `best.pl`)
Controls output filename for learned rules:
```bash
python main.py -g 1 -d kinship -p term0 -cur 1 -ft 0.3 -output_rules my_rules.pl
```

This allows:
- Debugging different extraction configurations
- Comparing multiple rule sets
- Custom naming for different experiments

## Data Flow

```
Training Data → Neural Model → Weight Extraction → Symbolic Rules → Filtering → Output File
     ↓              ↓                 ↓                  ↓             ↓            ↓
  train.pl      model/          nID.py            logic_program.pl  best.pl   Custom file
```

## Key Parameters

- **threshold**: Minimum weight value for rule activation (searched 0-1)
- **final_threshold**: Minimum accuracy for including rules in output (default: 0.3)
- **variable_depth**: Maximum number of variables in rule bodies (default: 1)

## Separate File Support

When using separate `train.pl` and `facts.pl`:
- `facts.pl` predicates define the domain for rule bodies
- `train.pl` predicates are learning targets only
- Target predicates excluded from background knowledge during extraction
- Prevents data leakage in rule learning

## Summary

The system uses a neural-symbolic approach where:
1. Neural network learns from relational data
2. Network weights encode logical rules
3. Extraction converts weights to interpretable Prolog rules
4. Rules are filtered by accuracy and written to output file
5. New flag allows custom output naming for debugging
