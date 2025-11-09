# Using Separate train.pl and facts.pl Files

## Overview

The DFORL system now supports loading data from two separate files:
- **facts.pl**: Background knowledge used for rule inference
- **train.pl**: Target edges to learn definitions for

This separation allows you to control which predicates are used in rule bodies and which facts are targets for learning.

## Key Features

1. **Background Knowledge (facts.pl)**: 
   - Contains predicates and facts that define the domain for rule generation
   - Only predicates from facts.pl are included in the body of learned rules
   - Used for computing rule inference

2. **Target Edges (train.pl)**:
   - Contains the target relations you want to learn definitions for
   - Facts in train.pl are NOT included in the background when computing rule inference
   - Used only as learning targets

## File Format

Both files use Prolog syntax with facts in the format:
```prolog
predicate(object1, object2).
```

Example facts.pl:
```prolog
parent(john, mary).
parent(john, bob).
male(john).
male(bob).
female(mary).
```

Example train.pl:
```prolog
father(john, mary).
father(john, bob).
```

In this example, the system will learn rules for the `father` relation using the predicates `parent`, `male`, and `female` from facts.pl, but will not include `father` facts from train.pl in the background knowledge.

## Directory Structure

Organize your dataset as follows:
```
deepDFOL/
  {dataset_name}/
    data/
      train.pl      # Target edges to learn
      facts.pl      # Background knowledge
```

## Usage

### Example with Kinship Dataset

1. **Prepare your data files**:
   
   Create `deepDFOL/kinship/data/train.pl` with target relations:
   ```prolog
   term0(person1, person2).
   term0(person3, person4).
   ...
   ```
   
   Create `deepDFOL/kinship/data/facts.pl` with background knowledge:
   ```prolog
   term1(person1, person5).
   term2(person2, person6).
   term3(person3, person7).
   ...
   ```

2. **Run DFORL**:
   ```bash
   python main.py -g 1 -d kinship -p term0 -cur 1 -ft 0.3
   ```

The system will automatically detect the presence of train.pl and facts.pl and use them appropriately.

## Backward Compatibility

If train.pl and facts.pl are not present, the system falls back to the original behavior of loading from a single .nl file:
```
deepDFOL/
  {dataset_name}/
    data/
      {dataset_name}.nl  # Original single-file format
```

## How It Works

1. **Detection**: The `main()` function in `data_generator.py` checks for the existence of train.pl and facts.pl
2. **Loading**: If both files exist, it uses `classifier_with_separate_files()` instead of the original `classifier()`
3. **Separation**: 
   - Background predicates are extracted only from facts.pl
   - Target facts are read from train.pl
   - The target predicate is excluded from the background knowledge domain
4. **Rule Generation**: Only predicates from facts.pl are considered when generating rule bodies

## Implementation Details

### New Function: `classifier_with_separate_files()`

Located in `data_generator.py`, this function:
- Takes separate paths for facts.pl and train.pl
- Loads background knowledge from facts.pl
- Loads target edges from train.pl
- Returns the same data structure as the original classifier for compatibility
- Ensures target predicates are not included in background relations

### Modified Function: `main()`

The main data generation function now:
- Checks for train.pl and facts.pl existence
- Calls `classifier_with_separate_files()` if both files are present
- Falls back to original `classifier()` for backward compatibility

## Benefits

1. **Clear Separation**: Explicitly separate learning targets from background knowledge
2. **Control**: Fine-grained control over which predicates are used in rule bodies
3. **Flexibility**: Can use the same background knowledge with different target relations
4. **Correctness**: Prevents target facts from being used in rule inference, avoiding data leakage

## Example: Learning Kinship Relations

Suppose you want to learn the `father` relation using parent relationships and gender information:

**facts.pl** (background knowledge):
```prolog
parent(john, mary).
parent(john, bob).
parent(susan, mary).
parent(susan, bob).
male(john).
male(bob).
female(mary).
female(susan).
```

**train.pl** (target to learn):
```prolog
father(john, mary).
father(john, bob).
```

The system will learn rules like:
```prolog
father(X, Y) :- parent(X, Y), male(X).
```

Note that the `father` facts from train.pl are not available during rule generation, ensuring the learned rules are based only on the background knowledge.

## Testing

The implementation has been tested with the kinship dataset:
- 24 background predicates loaded from facts.pl
- 9,401 background facts
- 211 target facts from train.pl
- Target predicate correctly excluded from background
- Variable objects configured correctly

## Troubleshooting

**Q: The system is still using the old single-file format**

A: Make sure both `train.pl` and `facts.pl` exist in `deepDFOL/{dataset}/data/`. If either file is missing, the system will fall back to loading from `{dataset}.nl`.

**Q: My target predicate appears in facts.pl**

A: That's okay! The system will still separate them correctly. Facts with the target predicate in facts.pl will be used as background knowledge, while facts in train.pl will be used as learning targets.

**Q: I get an error about missing files**

A: Check that your directory structure matches `deepDFOL/{dataset}/data/` and that your files have the `.pl` extension.
