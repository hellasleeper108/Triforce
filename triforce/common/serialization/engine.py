import inspect
import json
import types
import io
import sys
import textwrap
import traceback
import builtins

class JobSerializer:
    @staticmethod
    def serialize(func, args=None, kwargs=None):
        """
        Converts a Python function and its arguments into a portable JSON payload.
        Warning: This transmits source code. The function must be self-contained.
        """
        if not isinstance(func, (types.FunctionType, types.MethodType)):
            raise ValueError("Target must be a function or method")
            
        source = inspect.getsource(func)
        # Handle indented inner functions by stripping common whitespace
        source = textwrap.dedent(source)
        
        return json.dumps({
            "code": source,
            "entrypoint": func.__name__,
            "args": args or [],
            "kwargs": kwargs or {}
        })

    @staticmethod
    def deserialize(payload_str):
        """
        Reconstructs the job definition from JSON.
        """
        return json.loads(payload_str)

class SandboxError(Exception):
    pass

class JobSandbox:
    def __init__(self, allow_builtins=True):
        self.allow_builtins = allow_builtins

    def _get_restricted_globals(self):
        """
        Creates a restricted global namespace.
        Removes dangerous builtins like 'open', 'exec', 'eval', '__import__'.
        """
        safe_builtins = {}
        
        if self.allow_builtins:
            # explicit allow-list of safe builtins
            for name in [
                "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", 
                "callable", "chr", "complex", "dict", "divmod", "enumerate", "filter", 
                "float", "format", "frozenset", "getattr", "hasattr", "hash", "hex", 
                "id", "int", "isinstance", "issubclass", "iter", "len", "list", "map", 
                "max", "min", "next", "object", "oct", "ord", "pow", "print", "range", 
                "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", 
                "tuple", "type", "zip"
            ]:
                if hasattr(builtins, name):
                    safe_builtins[name] = getattr(builtins, name)

        return {
            "__builtins__": safe_builtins,
            "__name__": "__main__",
        }

    def execute(self, payload_str):
        """
        Executes the serialized job in a restricted environment.
        """
        try:
            job = JobSerializer.deserialize(payload_str)
        except json.JSONDecodeError:
            raise SandboxError("Invalid JSON payload")

        code = job.get("code")
        entrypoint = job.get("entrypoint")
        args = job.get("args", [])
        kwargs = job.get("kwargs", {})

        if not code or not entrypoint:
            raise SandboxError("Missing code or entrypoint in payload")

        # Capture Output
        capture_stdout = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = capture_stdout

        restricted_globals = self._get_restricted_globals()
        local_scope = {}

        try:
            # 1. Compile and Run the definition
            exec(code, restricted_globals, local_scope)
            
            # 2. Find the entrypoint
            if entrypoint not in local_scope:
                raise SandboxError(f"Entrypoint '{entrypoint}' not defined in job code")
            
            func = local_scope[entrypoint]
            
            # 3. Execution
            result = func(*args, **kwargs)
            
            return {
                "status": "SUCCESS",
                "result": result,
                "stdout": capture_stdout.getvalue()
            }
            
        except Exception as e:
            # traceback.print_exc() # Print to captured stdout/err
            return {
                "status": "FAILED",
                "error": str(e),
                "type": type(e).__name__,
                "stdout": capture_stdout.getvalue()
            }
        finally:
            sys.stdout = original_stdout
