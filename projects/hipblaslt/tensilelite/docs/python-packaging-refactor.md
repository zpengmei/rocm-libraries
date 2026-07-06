<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# TensileLite Python Packaging Refactor

## Problem

`projects/hipblaslt/tensilelite/Tensile` is the hipBLASLt fork of the rocBLAS
Tensile generator, but the Python package, import paths, and command names still
look like upstream Tensile:

- The distribution metadata currently publishes as `tensile`.
- The import namespace is the top-level `Tensile` package.
- Many in-repo callers execute files under `Tensile/bin` or patch `sys.path` so
  that `import Tensile` and `import rocisa` resolve from a source or build tree.
- CMake installs raw Python source trees for test artifacts instead of consuming
  an installed Python package.

This leaves no clean way for another project, such as GEKO, to declare a normal
Python dependency on the TensileLite assets. It also keeps the legacy rocBLAS
Tensile tool and the TensileLite fork ambiguous in both Python imports and
executable names.

## Goals

- Publish the TensileLite generator assets as a normal Python package.
- Use a canonical package name that cannot be confused with legacy rocBLAS
  Tensile.
- Provide import paths that identify the code as TensileLite.
- Replace source-tree executable files and `PYTHONPATH` mutation with supported
  module entry points and console scripts.
- Let downstream projects depend on TensileLite through standard packaging
  metadata.
- Keep the existing hipBLASLt CMake device-library build working throughout the
  migration.

## Non-goals

- Do not change code generation semantics.
- Do not change the C++ TensileLite host-library ABI.
- Do not rename generated kernel artifacts or library logic file formats.
- Do not require GEKO or other downstream projects to vendor the tensilelite
  source tree.

## Target Package Shape

Use these canonical names:

- Python distribution: `tensilelite`
- Python import namespace: `tensilelite`
- Required native dependency: `rocisa`

The package should install only the TensileLite namespace. A normal install of
`tensilelite` should not claim the top-level `Tensile` package, because that
recreates the conflict with legacy Tensile.

The package should expose stable public modules for downstream users and build
tools:

```python
import tensilelite
from tensilelite.create_library import main as create_library_main
from tensilelite.logic import main as logic_main
```

The first implementation can preserve the current CamelCase module filenames
internally if that reduces churn, but public docs and new callers should use the
`tensilelite` namespace. Over time, internal imports should be moved to relative
imports or to the new namespace so the implementation no longer depends on
`Tensile`.

## Command-Line Interface

Replace checked-in executable files under `Tensile/bin` with `pyproject.toml`
entry points. Suggested names:

- `tensilelite`
- `tensilelite-create-library`
- `tensilelite-logic`
- `tensilelite-merge-library`
- `tensilelite-retune-library`
- `tensilelite-liblogic-to-yaml`
- `tensilelite-generate-summations`

Each entry point should call a small `main(argv: Sequence[str] | None = None)`
function. The implementation should not read or rewrite `sys.argv` except at the
outermost CLI boundary.

During migration, legacy command names such as `TensileCreateLibrary` and
`TensileLogic` can remain as deprecated aliases in developer and CI workflows.
They should be thin console-script aliases, not source-tree scripts that recover
from import failures by editing `sys.path`.

## Build-System Integration

The hipBLASLt build should treat TensileLite as an installed Python package in
the build Python environment:

1. Install `tensilelite` and `rocisa` into the build venv, preferably
   editable during local development and as wheels in packaging tests.
2. Replace `HIPBLASLT_PYTHON_COMMAND` path injection with package execution.
3. Invoke device-library generation with:

   ```bash
   python -m tensilelite.create_library ...
   ```

4. Invoke logic validation with:

   ```bash
   python -m tensilelite.logic ...
   ```

5. Replace `HIPBLASLT_INSTALL_TENSILELITE_TEST_ARTIFACTS` raw source installs
   with installation of the package wheel and any required test data.

The current CMake call sites that execute `Tensile/bin/TensileLogic` or
`python -m Tensile.TensileCreateLibrary` should be converted after the new entry
points exist. This removes the build's dependence on a particular source-tree
layout.

## Package Data

The wheel must include all runtime assets that the generator reads from the
source tree today, including:

- `CustomKernels/*.s`
- `Source/*` files needed by generated clients and libraries
- `TensileCreateLibrary` and `TensileLogic` support data such as
  `known_bugs.yaml`
- CMake helper files that are still required by supported Python workflows

Code should access these files through `importlib.resources` instead of
constructing paths from `__file__` and assuming a checkout layout. Any path that
must remain user-visible should have an explicit API.

## Migration Plan

1. Add the new package metadata.
   Rename the distribution from `tensile` to `tensilelite`, add `rocisa` as a
   dependency, split runtime dependencies from dev/test tools, and add complete
   console-script entry points.

2. Add the new import namespace.
   Introduce `tensilelite` as the supported import path. Keep the implementation
   thin at first if needed, but make all new callers use the new namespace.

3. Convert internal and in-repo imports.
   Move implementation imports from `Tensile.*` to relative imports or
   `tensilelite.*`. Update tests, CMake Python invocations, and helper scripts.
   This is the main behavioral risk and should be covered by the existing
   characterization tests.

4. Deprecate legacy entry points.
   Keep `Tensile`, `TensileCreateLibrary`, and related command names only as
   warnings-backed compatibility aliases for a defined release window. Do not
   require those aliases for hipBLASLt's own build.

5. Stop installing the top-level `Tensile` package.
   Once in-repo and downstream callers have migrated, remove the `Tensile`
   package from the wheel. If source-tree compatibility is still needed for
   tests, keep it outside the installable package or move it to a separate
   compatibility package.

## Downstream Dependency Model

GEKO and other consumers should be able to declare:

```toml
[project]
dependencies = [
  "tensilelite>=5.0",
]
```

For monorepo or source-based development before publication, downstreams can use
a direct reference to the subdirectory:

```toml
tensilelite @ git+https://github.com/ROCm/rocm-libraries.git@develop#subdirectory=projects/hipblaslt/tensilelite
```

That dependency should provide importable APIs and command-line tools without
requiring GEKO to set `PYTHONPATH`, copy `Tensile/bin`, or know the hipBLASLt
repository layout.

## Validation

The refactor should be validated in both package and hipBLASLt build modes:

- Build a wheel with `python -m build`.
- Install the wheel into a clean venv and verify `import tensilelite`.
- Verify a clean venv does not import top-level `Tensile` from the new package
  once the compatibility window ends.
- Run console-script `--help` checks for every supported command.
- Run `tox -e unit` from `tensilelite`.
- Run a scoped hipBLASLt device-library build with `TENSILELITE_LOGIC_FILTER`.
- Run `scripts/run_tensile_logic_check.py` after it has been converted to the
  installed package API.
- Build or smoke-test a GEKO environment that depends on `tensilelite` without
  extra path configuration.

## Open Questions

- How long should legacy `Tensile*` command aliases remain available?
- Should a separate compatibility package provide the old top-level `Tensile`
  namespace for users that cannot migrate immediately?
- Should the ROCm binary package install Python wheels directly, or continue to
  place package contents under `share/hipblaslt` for test artifacts?
