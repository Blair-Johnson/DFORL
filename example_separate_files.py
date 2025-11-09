#!/usr/bin/env python
"""
Example script demonstrating the use of separate train.pl and facts.pl files

This example creates a simple dataset and shows how the new loading mechanism works.
"""

import os
import sys

def create_example_dataset():
    """Create a simple example dataset with train.pl and facts.pl"""
    
    # Create directory structure
    dataset_dir = "deepDFOL/example/data"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create facts.pl with background knowledge
    facts_content = """% Background knowledge: parent and gender relationships
parent(john, mary).
parent(john, bob).
parent(susan, mary).
parent(susan, bob).
male(john).
male(bob).
female(mary).
female(susan).
"""
    
    with open(os.path.join(dataset_dir, "facts.pl"), "w") as f:
        f.write(facts_content)
    
    # Create train.pl with target relation to learn
    train_content = """% Target relation to learn: father
father(john, mary).
father(john, bob).
"""
    
    with open(os.path.join(dataset_dir, "train.pl"), "w") as f:
        f.write(train_content)
    
    print("✓ Created example dataset:")
    print(f"  - {dataset_dir}/facts.pl (background knowledge)")
    print(f"  - {dataset_dir}/train.pl (target relation)")
    print()
    print("The system will learn rules for 'father' using predicates")
    print("'parent', 'male', and 'female' from facts.pl")
    print()
    print("Expected learned rule:")
    print("  father(X, Y) :- parent(X, Y), male(X).")
    print()
    print("Note: The 'father' facts in train.pl will NOT be available")
    print("during rule generation, ensuring clean separation.")

if __name__ == "__main__":
    create_example_dataset()
