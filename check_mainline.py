#!/usr/bin/env python3
"""Simple syntax and structure check for mainline.py"""

import ast
import sys
from pathlib import Path

def check_mainline_structure():
    """Check that mainline.py has correct structure and method signatures."""
    
    mainline_path = Path("src/mainline.py")
    
    with open(mainline_path, 'r') as f:
        content = f.read()
    
    # Parse the AST
    tree = ast.parse(content)
    
    found_methods = {}
    
    # Find classes and their methods
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            found_methods[class_name] = []
            
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    found_methods[class_name].append(method.name)
    
    print("Found classes and methods:")
    for class_name, methods in found_methods.items():
        print(f"  {class_name}: {methods}")
    
    # Check for specific requirements
    chat_methods = found_methods.get('ChatChainModel', [])
    
    required_methods = ['__init__', '_build_document', '_build_chain', 'handle_prompt', 'get_layout']
    missing_methods = [m for m in required_methods if m not in chat_methods]
    
    if missing_methods:
        print(f"❌ Missing methods in ChatChainModel: {missing_methods}")
        return False
    
    print("✅ All required methods found in ChatChainModel")
    
    # Check if handle_prompt uses the prompt parameter
    handle_prompt_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'handle_prompt':
            handle_prompt_node = node
            break
    
    if handle_prompt_node:
        # Check if 'prompt' parameter is used in the function body
        uses_prompt = False
        for node in ast.walk(handle_prompt_node):
            if isinstance(node, ast.Name) and node.id == 'prompt':
                uses_prompt = True
                break
        
        if uses_prompt:
            print("✅ handle_prompt method uses the prompt parameter")
        else:
            print("❌ handle_prompt method does not use the prompt parameter")
            return False
    
    # Check if _build_chain returns compiled graph
    build_chain_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_build_chain':
            build_chain_node = node
            break
    
    if build_chain_node:
        # Look for compile() call
        has_compile = False
        for node in ast.walk(build_chain_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'compile':
                has_compile = True
                break
        
        if has_compile:
            print("✅ _build_chain method calls compile()")
        else:
            print("❌ _build_chain method does not call compile()")
            return False
    
    return True

if __name__ == "__main__":
    success = check_mainline_structure()
    if success:
        print("\n🎉 All structure checks passed!")
    else:
        print("\n❌ Some checks failed!")
        sys.exit(1)