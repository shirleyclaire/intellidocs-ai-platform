import sys
import os

# Ensure the platform root is in the path
workspace = r"C:\Users\Shirley Claire\Desktop\Shirley\Projects\intellidocs-ai-platform\desicrew_platform"
if workspace not in sys.path:
    sys.path.insert(0, workspace)

import pandas as pd
from task1_excel_agent.agent import build_agent, run_query

def main():
    # Create a minimal test dataframe
    df = pd.DataFrame({
        'product': ['Widget A', 'Widget B', 'Widget C'],
        'warehouse': ['WH1', 'WH2', 'WH1'],
        'stock': [100, 50, 200],
        'unit_price': [10.0, 25.0, 5.0]
    })

    print("Building agent...")
    try:
        agent = build_agent(df)
        print("Agent built successfully!\n")
        print("-" * 50)
        
        # Define our test scenarios
        test_queries = [
            "What is the total stock across all warehouses?", 
            "What is the total inventory value? (Calculate stock multiplied by unit price for all items and sum it up)",
            "Which warehouse has the highest total stock?"
        ]

        # Run each query and print results
        for i, query in enumerate(test_queries, 1):
            print(f"TEST {i}: {query}")
            result = run_query(agent, query)
            
            print(f"Answer: {result['answer']}")
            print(f"Code Executed:\n{result['code']}")
            if result['error']:
                print(f"Error: {result['error']}")
                
            print("-" * 50)

    except Exception as e:
        print(f"FAILED TO BUILD AGENT OR RUN QUERY: {e}")

if __name__ == "__main__":
    main()