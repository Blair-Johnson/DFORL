#!/usr/bin/env python
"""
Test script for the new classifier_with_separate_files function
"""
import os
import sys
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import the classifier function
from data_generator import classifier_with_separate_files

def test_kinship_separate_files():
    """Test the new classifier with kinship dataset split into train.pl and facts.pl"""
    
    facts_path = 'deepDFOL/kinship/data/facts.pl'
    train_path = 'deepDFOL/kinship/data/train.pl'
    output_path = 'deepDFOL/kinship/data/term0'
    target_relation = 'term0'
    variable_number = 3  # variable_depth + 2, where variable_depth = 1
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    logging.info("Testing classifier_with_separate_files...")
    logging.info(f"Facts path: {facts_path}")
    logging.info(f"Train path: {train_path}")
    logging.info(f"Target relation: {target_relation}")
    
    # Call the new classifier
    variable_objects, relation_names, relation, arity_relation, target_arity = classifier_with_separate_files(
        facts_path, train_path, variable_number, output_path, target_relation
    )
    
    logging.info("=" * 80)
    logging.info("RESULTS:")
    logging.info("=" * 80)
    logging.info(f"Target arity: {target_arity}")
    logging.info(f"Number of background predicates: {len(relation_names)}")
    logging.info(f"Background predicates: {relation_names}")
    logging.info(f"Target relation '{target_relation}' in background: {target_relation in relation_names}")
    
    # Verify that term0 is NOT in the background relations (as expected)
    if target_relation not in relation_names:
        logging.info("✓ PASS: Target relation correctly excluded from background predicates")
    else:
        logging.warning("✗ FAIL: Target relation should not be in background predicates")
    
    # Count facts in background
    total_facts = sum(len(relation[rel]) for rel in relation)
    logging.info(f"Total background facts: {total_facts}")
    
    # Check variable objects
    logging.info(f"Variable object counts: {[(k, len(v)) for k, v in variable_objects.items()]}")
    
    # Verify output files were created
    if os.path.exists(output_path + '/all_relation_dic.dt'):
        logging.info("✓ all_relation_dic.dt created")
    if os.path.exists(output_path + '/relation_entities.dt'):
        logging.info("✓ relation_entities.dt created")
    if os.path.exists(output_path + '/pro.dt'):
        logging.info("✓ pro.dt created")
    
    return True

if __name__ == '__main__':
    try:
        test_kinship_separate_files()
        print("\n" + "=" * 80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)
    except Exception as e:
        logging.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)
