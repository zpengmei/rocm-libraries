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
entry points. The only new public command should be `tensilelite`, with
subcommands for the existing tools:

- `tensilelite run`
- `tensilelite create-library`
- `tensilelite logic`
- `tensilelite merge-library`
- `tensilelite retune-library`
- `tensilelite update-library`
- `tensilelite liblogic-to-yaml`
- `tensilelite generate-summations`

Do not add new executable-per-tool names such as `tensilelite-create-library`.
Those names add a second public command family without solving a compatibility
problem.

The new package metadata should have one primary entry point plus legacy aliases:

```toml
[project.scripts]
tensilelite = "tensilelite.cli:main"
Tensile = "tensilelite.cli.compat:tensile"
TensileBenchmarkCluster = "tensilelite.cli.compat:benchmark_cluster"
TensileCreateLibrary = "tensilelite.cli.compat:create_library"
TensileGenerateSummations = "tensilelite.cli.compat:generate_summations"
TensileLibLogicToYaml = "tensilelite.cli.compat:liblogic_to_yaml"
TensileLogic = "tensilelite.cli.compat:logic"
TensileMergeLibrary = "tensilelite.cli.compat:merge_library"
TensileRetuneLibrary = "tensilelite.cli.compat:retune_library"
TensileUpdateLibrary = "tensilelite.cli.compat:update_library"
```

The package should also provide `tensilelite/__main__.py` so
`python -m tensilelite ...` uses the same dispatcher as the installed
`tensilelite` console script.

Each compatibility alias must flow through the new CLI dispatcher instead of
calling the old implementation directly. The alias should prepend the matching
subcommand and pass a `compat_path` value into the new main path:

```python
def create_library(argv: Sequence[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    return main(
        ["create-library", *args],
        compat_path=("TensileCreateLibrary", "tensilelite create-library"),
    )
```

The shared main/subcommand functions should print a deprecation warning when
`compat_path` is set:

```text
TensileCreateLibrary is deprecated and will be removed in a future release.
Use `tensilelite create-library` instead.
```

This keeps warning text, exit behavior, logging, and argument parsing on the new
code path while preserving the old command names for existing scripts.

## Build-System Integration

The hipBLASLt build should treat TensileLite as an installed Python package in
the build Python environment:

1. Install `tensilelite` and `rocisa` into the build venv, preferably
   editable during local development and as wheels in packaging tests.
2. Replace `HIPBLASLT_PYTHON_COMMAND` path injection with package execution.
3. Invoke device-library generation with:

   ```bash
   python -m tensilelite create-library ...
   ```

4. Invoke logic validation with:

   ```bash
   python -m tensilelite logic ...
   ```

5. Replace `HIPBLASLT_INSTALL_TENSILELITE_TEST_ARTIFACTS` raw source installs
   with installation of the package wheel and any required test data.

The current CMake call sites that execute `Tensile/bin/TensileLogic` or
`python -m Tensile.TensileCreateLibrary` should be converted after the new entry
points exist. This removes the build's dependence on a particular source-tree
layout.

## Package Data

The package-data contract should support the features currently reachable from
`Tensile/bin` without copying the entire current source tree:

- Library logic YAML generation in the `tensilelite run` pipeline primarily
  needs Python modules and its input benchmark data. It does not need static
  source headers, `known_bugs.yaml`, or CMake helper files just to analyze
  benchmark CSV/YAML data and write `3_LibraryLogic/*.yaml`.
- Library generation (`tensilelite create-library`) needs the static source
  headers copied today by `copyStaticFiles`: `TensileTypes.h`,
  `tensile_bfloat16.h`, `tensile_float8_bfloat8.h`, `KernelHeader.h`,
  `ReductionTemplate.h`, and `memory_gfx.h`.
- `CustomKernels/*.s` must be included if packaged TensileLite supports custom
  kernel workflows. Benchmark configs can request custom kernels with
  `CustomKernels`, logic files can reference `CustomKernelName`, and codegen
  reads the matching assembly file from `CustomKernels/`.
- `known_bugs.yaml` should be included if the packaged `tensilelite logic
  --check-all` command is expected to provide the same default documented-skip
  behavior as the hipBLASLt CMake validation gate. It is not needed for
  producing LibraryLogic YAMLs or code objects.
- CMake helper files from `Tensile/Source` should not be included by default for
  the `Tensile/bin` feature boundary. They appear to be legacy support for an
  older CMake-based client/source workflow; in this tree, `Tensile/Source` has
  no `CMakeLists.txt`, and the active client path expects a prebuilt client.

Package-data decisions:

- Include the static source headers required by `copyStaticFiles`.
- Include `CustomKernels/*.s`; custom kernel support is part of the current
  command surface and codegen reads these files by name.
- Include `TensileLogic/known_bugs.yaml` only as validation support for
  `tensilelite logic --check-all`; it is not a generation input.
- Exclude `Tensile/Source/EnableWarnings.cmake` and
  `Tensile/Source/FindOpenCL.cmake` from the wheel unless a supported packaged
  workflow is found that still uses them. A repository scan found no tracked
  callers outside the files themselves.

Code should access these files through `importlib.resources` instead of
constructing paths from `__file__` and assuming a checkout layout. Any path that
must remain user-visible should have an explicit API.

## Legacy Import Compatibility

The canonical `tensilelite` wheel should not install a top-level `Tensile`
package once the new namespace is ready. A same-wheel shim would make every
`tensilelite` install claim the legacy `Tensile` namespace, recreate the naming
conflict this refactor is meant to remove, and keep APIs such as
`Tensile.ROOT_PATH`, `Tensile.SOURCE_PATH`, and `Tensile.CUSTOM_KERNEL_PATH`
alive by default.

If external users need a short-lived transition path for `import Tensile.*`,
ship it as a separate opt-in compatibility distribution, for example
`tensilelite-tensile-compat`. That package should:

- depend on a tightly matched `tensilelite` version;
- install the top-level `Tensile` namespace only in the compatibility package;
- warn visibly on `import Tensile`, preferably with a `FutureWarning`-based
  custom warning because `DeprecationWarning` is often hidden;
- include a specific removal release in the warning text;
- document that it is mutually exclusive with any other package that owns the
  `Tensile` import namespace.

The compatibility package must avoid loading implementation modules twice. A
naive `__path__` shim can make `Tensile.Common.DataType` and
`tensilelite.Common.DataType` become different module objects, splitting module
globals, class identities, caches, and `globalParameters`. If an import
compatibility package is required, it should alias through `sys.modules` or an
import hook so legacy `Tensile.*` imports resolve to the same module objects as
the new `tensilelite.*` imports.

The main `tensilelite` wheel should still keep temporary `Tensile*`
console-script aliases during the transition. Console-script aliases do not
claim the Python `Tensile` namespace and can safely route through
`tensilelite.cli` with `compat_path` warnings.

## Migration Plan

1. Add the new package metadata.
   Rename the distribution from `tensile` to `tensilelite`, add `rocisa` as a
   dependency, split runtime dependencies from dev/test tools, add the
   `tensilelite` console script, and add legacy `Tensile*` aliases that dispatch
   through the new CLI with `compat_path`.

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
   require those aliases for hipBLASLt's own build, and do not add new
   `tensilelite-*` executable aliases.

5. Replace implicit source-tree asset reads.
   Move the static headers, `CustomKernels/*.s`, and `known_bugs.yaml` behind
   explicit resource-access helpers. As part of this step, verify that
   `Tensile/Source/EnableWarnings.cmake` and `Tensile/Source/FindOpenCL.cmake`
   remain unused; if so, leave them out of package data and consider removing
   them from the source tree in a follow-up cleanup.

6. Stop installing the top-level `Tensile` package.
   Once in-repo and downstream callers have migrated, remove the `Tensile`
   package from the canonical wheel. `pip install tensilelite` should make
   `import tensilelite` work and `import Tensile` fail. If a short transition
   for legacy Python imports is required, use a separate opt-in compatibility
   package as described above; do not keep `Tensile` as a shim inside the main
   wheel.

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
- Run `tensilelite --help` and subcommand `--help` checks for every supported
  mode.
- Run compatibility alias smoke tests and verify each one prints the deprecation
  warning with the replacement `tensilelite <subcommand>` command.
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
