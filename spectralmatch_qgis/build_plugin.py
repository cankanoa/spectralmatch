import inspect
import json
import importlib
import os
from types import FunctionType
from typing import List

def generate_function_headers(
    package_name="spectralmatch",
    output_file="spectralmatch_qgis/function_headers.json",
    exclude_functions: List[str] = None,
    exclude_modules: List[str] = ["spectralmatch.handlers"],
    exclude_internal_functions: bool = True
):
    exclude_functions = set(exclude_functions or [])
    output = []

    exclude_functions = set(exclude_functions or [])
    exclude_modules = set(exclude_modules or [])
    output = []

    def walk_module(module, prefix):
        if any(prefix.startswith(excl) for excl in exclude_modules):
            return

        for name in dir(module):
            if name in exclude_functions or (exclude_internal_functions and name.startswith("_")):
                continue
            try:
                obj = getattr(module, name)
            except Exception:
                continue
            if inspect.ismodule(obj) and obj.__package__ and obj.__package__.startswith(package_name):
                walk_module(obj, f"{prefix}.{name}")
            elif isinstance(obj, FunctionType) and obj.__module__ == module.__name__:
                if "." in obj.__qualname__:
                    continue

                docstring = inspect.getdoc(obj) or ""
                sig = inspect.signature(obj)
                params = []
                for param in sig.parameters.values():
                    annotation = None
                    if param.annotation is not param.empty:
                        ann = repr(param.annotation)
                        if ann.startswith("typing."):
                            annotation = ann.replace("typing.", "")
                        elif ann.startswith("<class '") and ann.endswith("'>"):
                            annotation = ann[8:-2]
                        else:
                            annotation = ann

                    param_type = "folder" if param.name in {"input_images", "output_images"} else "string"

                    params.append({
                        "name": param.name,
                        "display_name": param.name.replace("_", " ").capitalize(),
                        "kind": str(param.kind),
                        "default": repr(param.default) if param.default is not param.empty else None,
                        "annotation": annotation,
                        "param_type": param_type,
                    })

                output.append({
                    "function": f"{prefix}.{name}",
                    "docstring": docstring,
                    "parameters": params
                })

    pkg = importlib.import_module(package_name)
    walk_module(pkg, package_name)
    if not output: raise RuntimeError(f"No function headers found in package '{package_name}'. Check exclusions, installation, or package contents.")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    return output_file


def generate_requirements_txt(
    input_toml_path="pyproject.toml",
    output_txt_path="spectralmatch_qgis/requirements.txt",
):
    with open(input_toml_path, "r") as f:
        lines = f.readlines()

    in_project_section = False
    in_dependencies_list = False
    deps = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("[project]"):
            in_project_section = True
            continue
        elif stripped.startswith("[") and not stripped.startswith("[project."):
            in_project_section = False
            in_dependencies_list = False
            continue

        if in_project_section and stripped.startswith("dependencies"):
            in_dependencies_list = True
            continue

        if in_dependencies_list:
            if stripped.startswith("]") or stripped.startswith("["):
                break
            if stripped:
                dep = stripped.strip().rstrip(",").strip('"').strip("'")
                deps.append(dep)

    with open(output_txt_path, "w") as f:
        for dep in deps:
            f.write(dep + "\n")

if __name__ == "__main__":
    generate_function_headers()
    generate_requirements_txt()
