#!/usr/bin/env python
"""
Validation script to verify the data loading pipeline modifications work correctly
"""

import os
import sys
import logging
from unittest.mock import MagicMock

# Mock tensorflow since it's not critical for validation
sys.modules['tensorflow'] = MagicMock()
sys.modules['tensorflow.train'] = MagicMock()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def validate_directory_structure():
    """Validate that datasets are organized correctly"""
    logging.info("Validating directory structure...")
    
    datasets = ['kinship', 'nations', 'umls', 'UW-CSE', 'fb15k_237', 'wn18', 'wn18rr', 'length']
    missing = []
    
    for dataset in datasets:
        path = f'deepDFOL/{dataset}/data'
        if not os.path.exists(path):
            missing.append(dataset)
    
    if missing:
        logging.error(f"Missing directories for: {missing}")
        return False
    
    logging.info(f"✓ All {len(datasets)} dataset directories exist")
    return True

def validate_kinship_split():
    """Validate the kinship dataset split into train.pl and facts.pl"""
    logging.info("Validating kinship dataset split...")
    
    facts_path = 'deepDFOL/kinship/data/facts.pl'
    train_path = 'deepDFOL/kinship/data/train.pl'
    
    if not os.path.exists(facts_path):
        logging.error(f"Missing: {facts_path}")
        return False
    
    if not os.path.exists(train_path):
        logging.error(f"Missing: {train_path}")
        return False
    
    # Count lines
    with open(facts_path, 'r') as f:
        facts_lines = len([l for l in f.readlines() if l.strip() and not l.strip().startswith('%')])
    
    with open(train_path, 'r') as f:
        train_lines = len([l for l in f.readlines() if l.strip() and not l.strip().startswith('%')])
    
    logging.info(f"  facts.pl: {facts_lines} facts")
    logging.info(f"  train.pl: {train_lines} facts")
    
    # Verify train.pl only contains term0
    with open(train_path, 'r') as f:
        train_predicates = set()
        for line in f:
            if line.strip() and not line.strip().startswith('%'):
                pred = line.split('(')[0]
                train_predicates.add(pred)
    
    if train_predicates != {'term0'}:
        logging.error(f"train.pl should only contain term0, but has: {train_predicates}")
        return False
    
    # Verify facts.pl doesn't contain term0
    with open(facts_path, 'r') as f:
        has_term0 = any('term0(' in line for line in f)
    
    if has_term0:
        logging.error("facts.pl should not contain term0")
        return False
    
    logging.info("✓ Kinship dataset correctly split")
    return True

def validate_classifier_function():
    """Validate the new classifier function works"""
    logging.info("Validating classifier_with_separate_files function...")
    
    try:
        from data_generator import classifier_with_separate_files
        logging.info("✓ classifier_with_separate_files imported successfully")
    except ImportError as e:
        logging.error(f"Failed to import classifier_with_separate_files: {e}")
        return False
    
    # Test with kinship dataset
    try:
        facts_path = 'deepDFOL/kinship/data/facts.pl'
        train_path = 'deepDFOL/kinship/data/train.pl'
        output_path = '/tmp/test_kinship'
        target_relation = 'term0'
        variable_number = 3
        
        os.makedirs(output_path, exist_ok=True)
        
        result = classifier_with_separate_files(
            facts_path, train_path, variable_number, output_path, target_relation
        )
        
        variable_objects, relation_names, relation, arity_relation, target_arity = result
        
        # Validate results
        if target_relation in relation_names:
            logging.error("Target relation should not be in background predicates")
            return False
        
        if len(relation_names) != 24:
            logging.error(f"Expected 24 background predicates, got {len(relation_names)}")
            return False
        
        if target_arity != 2:
            logging.error(f"Expected target arity 2, got {target_arity}")
            return False
        
        logging.info(f"✓ Classifier function works correctly")
        logging.info(f"  - {len(relation_names)} background predicates")
        logging.info(f"  - Target arity: {target_arity}")
        logging.info(f"  - Target excluded from background: {target_relation not in relation_names}")
        
        return True
        
    except Exception as e:
        logging.error(f"Classifier function failed: {e}", exc_info=True)
        return False

def validate_backward_compatibility():
    """Validate that original .nl file loading still works"""
    logging.info("Validating backward compatibility...")
    
    try:
        from data_generator import classifier
        
        # Test with a dataset that has .nl file - use kinship which we know works
        nl_path = 'deepDFOL/kinship/data/kinship.nl'
        output_path = '/tmp/test_backward_compat'
        
        os.makedirs(output_path, exist_ok=True)
        
        result = classifier(nl_path, 3, output_path, 'term1')
        
        if result:
            logging.info("✓ Backward compatibility maintained - original classifier works")
            return True
        
    except Exception as e:
        logging.error(f"Backward compatibility test failed: {e}", exc_info=True)
        return False
    
    return True

def validate_documentation():
    """Validate that documentation files exist"""
    logging.info("Validating documentation...")
    
    docs = [
        'SEPARATE_FILES_GUIDE.md',
        'example_separate_files.py',
        'README.md'
    ]
    
    missing = []
    for doc in docs:
        if not os.path.exists(doc):
            missing.append(doc)
    
    if missing:
        logging.error(f"Missing documentation files: {missing}")
        return False
    
    logging.info(f"✓ All {len(docs)} documentation files exist")
    return True

def main():
    """Run all validations"""
    print("=" * 80)
    print("DFORL Data Loading Pipeline Validation")
    print("=" * 80)
    print()
    
    tests = [
        ("Directory Structure", validate_directory_structure),
        ("Kinship Dataset Split", validate_kinship_split),
        ("Classifier Function", validate_classifier_function),
        ("Backward Compatibility", validate_backward_compatibility),
        ("Documentation", validate_documentation),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nTest: {name}")
        print("-" * 80)
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logging.error(f"Test failed with exception: {e}", exc_info=True)
            results.append((name, False))
        print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    print("=" * 80)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        return 0
    else:
        print("SOME TESTS FAILED ✗")
        print("=" * 80)
        return 1

if __name__ == '__main__':
    sys.exit(main())
