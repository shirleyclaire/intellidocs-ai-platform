import sys
import os

workspace = r"C:\Users\Shirley Claire\Desktop\Shirley\Projects\intellidocs-ai-platform\desicrew_platform"
if workspace not in sys.path:
    sys.path.insert(0, workspace)

def test_imports():
    modules = [
        "shared.llm",
        "shared.utils",
        "shared.prompts",
        "shared.embeddings",
        "shared.vector_store",
        "shared.ocr"
    ]
    all_passed = True
    for mod in modules:
        try:
            __import__(mod)
            print(f"OK: {mod}")
        except Exception as e:
            print(f"FAIL: {mod} - {e}")
            all_passed = False
            
    if not all_passed:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")

if __name__ == "__main__":
    test_imports()
