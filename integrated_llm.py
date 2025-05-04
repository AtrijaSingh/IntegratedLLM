import ctypes
import os
import sys

# --- Adjust the DLL path ---
# Use Debug build DLL
_dll_path = r'C:/Users/atrij/Workspace/IntegratedLLM/combined_approach/NewCleanApproach_Integrated_LLM/dlls/Debug/integrated_llm.dll'

# Use Release build DLL
#_dll_path = r'C:/Users/atrij/Workspace/IntegratedLLM/combined_approach/NewCleanApproach_Integrated_LLM/dlls/Release/integrated_llm.dll'

print("Loading DLL from:", _dll_path)

# Load the compiled C++ DLL
_llm = ctypes.CDLL(_dll_path)

# Define function signatures
_llm.init.argtypes = [ctypes.c_char_p]
_llm.init.restype = ctypes.c_int

_llm.add_knowledge.argtypes = [ctypes.c_char_p]
_llm.add_knowledge.restype = None

_llm.load_knowledge_file.argtypes = [ctypes.c_char_p]
_llm.load_knowledge_file.restype = None

_llm.query.argtypes = [ctypes.c_char_p]
_llm.query.restype = ctypes.c_void_p

_llm.free_response.argtypes = [ctypes.c_void_p]
_llm.free_response.restype = None

# --- Internal initialization flag ---
_initialized = False

# --- Python wrappers for C++ API ---

def init(model_path):
    """Initialize the LLM with the model path."""
    global _initialized
    if not _initialized:
        ret = _llm.init(model_path.encode())
        if ret != 0:
            raise RuntimeError(f"Failed to initialize LLM (code {ret})")
        _initialized = True

def add_knowledge(doc):
    """Add a knowledge string into the memory-based knowledge base."""
    if not _initialized:
        raise RuntimeError("LLM not initialized. Call init() first.")
    _llm.add_knowledge(doc.encode())

def load_knowledge_file(filepath):
    """Load knowledge entries from a text file."""
    if not _initialized:
        raise RuntimeError("LLM not initialized. Call init() first.")
    _llm.load_knowledge_file(filepath.encode())

def query(prompt):
    """Query the model with a prompt and get the generated response."""
    if not _initialized:
        raise RuntimeError("LLM not initialized. Call init() first.")
    
    resp_ptr = _llm.query(prompt.encode())
    if not resp_ptr:
        return None

    # Read C-string result
    resp = ctypes.cast(resp_ptr, ctypes.c_char_p).value.decode()

    # Free memory allocated by C++
    _llm.free_response(resp_ptr)

    return resp
