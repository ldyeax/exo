import hashlib
import importlib.machinery
import importlib.util
from pathlib import Path


def calculate_sglang_kt_artifact_build_id(
    module_name: str,
    required_native_fragment: str,
    additional_native_module_names: tuple[str, ...] = (),
) -> str | None:
    """Hash installed kernel sources and native extensions without absolute paths."""

    try:
        module_spec = importlib.util.find_spec(module_name)
        if module_spec is None:
            return None
        roots: list[Path] = []
        if module_spec.submodule_search_locations:
            roots.extend(
                Path(location).resolve()
                for location in module_spec.submodule_search_locations
            )
        elif module_spec.origin is not None:
            roots.append(Path(module_spec.origin).resolve())
        if not roots:
            return None

        native_suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
        source_suffixes = (
            ".c",
            ".cc",
            ".cpp",
            ".cu",
            ".cuh",
            ".h",
            ".hpp",
            ".json",
            ".py",
            ".pyi",
        )
        artifact_files: list[tuple[str, Path]] = []
        native_names: list[str] = []
        for root_index, root in enumerate(sorted(roots, key=str)):
            candidates = (root,) if root.is_file() else root.rglob("*")
            for candidate in candidates:
                if not candidate.is_file() or "__pycache__" in candidate.parts:
                    continue
                candidate_name = candidate.name
                is_native = candidate_name.endswith(native_suffixes)
                if not is_native and not candidate_name.endswith(source_suffixes):
                    continue
                relative_name = (
                    candidate.name
                    if root.is_file()
                    else candidate.relative_to(root).as_posix()
                )
                artifact_files.append(
                    (f"package:{root_index}:{relative_name}", candidate.resolve())
                )
                if is_native:
                    native_names.append(relative_name)

        for additional_module_name in additional_native_module_names:
            try:
                additional_spec = importlib.util.find_spec(additional_module_name)
            except ModuleNotFoundError:
                additional_spec = None
            if additional_spec is None or additional_spec.origin is None:
                continue
            additional_path = Path(additional_spec.origin).resolve()
            if not additional_path.is_file() or not additional_path.name.endswith(
                native_suffixes
            ):
                continue
            if any(
                artifact_path == additional_path
                for _relative_name, artifact_path in artifact_files
            ):
                native_names.append(f"{additional_module_name}:{additional_path.name}")
                continue
            artifact_files.append(
                (
                    f"module:{additional_module_name}:{additional_path.name}",
                    additional_path,
                )
            )
            native_names.append(f"{additional_module_name}:{additional_path.name}")

        if not artifact_files or not any(
            required_native_fragment in name for name in native_names
        ):
            return None

        digest = hashlib.sha256()
        digest.update(b"exo-sglang-kt-artifact-v1\0")
        for relative_name, artifact_path in sorted(artifact_files):
            encoded_name = relative_name.encode("utf-8")
            digest.update(len(encoded_name).to_bytes(8, "big"))
            digest.update(encoded_name)
            with artifact_path.open("rb") as artifact_file:
                while chunk := artifact_file.read(1024 * 1024):
                    digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


SGLANG_KT_ARTIFACT_BUILD_ID_FUNCTION_SOURCE = r"""
def calculate_sglang_kt_artifact_build_id(
    module_name,
    required_native_fragment,
    additional_native_module_names=(),
):
    try:
        module_spec = importlib.util.find_spec(module_name)
        if module_spec is None:
            return None
        roots = []
        if module_spec.submodule_search_locations:
            roots.extend(
                pathlib.Path(location).resolve()
                for location in module_spec.submodule_search_locations
            )
        elif module_spec.origin is not None:
            roots.append(pathlib.Path(module_spec.origin).resolve())
        if not roots:
            return None

        native_suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
        source_suffixes = (
            ".c",
            ".cc",
            ".cpp",
            ".cu",
            ".cuh",
            ".h",
            ".hpp",
            ".json",
            ".py",
            ".pyi",
        )
        artifact_files = []
        native_names = []
        for root_index, root in enumerate(sorted(roots, key=str)):
            candidates = (root,) if root.is_file() else root.rglob("*")
            for candidate in candidates:
                if not candidate.is_file() or "__pycache__" in candidate.parts:
                    continue
                candidate_name = candidate.name
                is_native = candidate_name.endswith(native_suffixes)
                if not is_native and not candidate_name.endswith(source_suffixes):
                    continue
                relative_name = (
                    candidate.name
                    if root.is_file()
                    else candidate.relative_to(root).as_posix()
                )
                artifact_files.append(
                    (f"package:{root_index}:{relative_name}", candidate.resolve())
                )
                if is_native:
                    native_names.append(relative_name)

        for additional_module_name in additional_native_module_names:
            try:
                additional_spec = importlib.util.find_spec(additional_module_name)
            except ModuleNotFoundError:
                additional_spec = None
            if additional_spec is None or additional_spec.origin is None:
                continue
            additional_path = pathlib.Path(additional_spec.origin).resolve()
            if not additional_path.is_file() or not additional_path.name.endswith(
                native_suffixes
            ):
                continue
            if any(
                artifact_path == additional_path
                for _relative_name, artifact_path in artifact_files
            ):
                native_names.append(
                    f"{additional_module_name}:{additional_path.name}"
                )
                continue
            artifact_files.append(
                (
                    f"module:{additional_module_name}:{additional_path.name}",
                    additional_path,
                )
            )
            native_names.append(f"{additional_module_name}:{additional_path.name}")

        if not artifact_files or not any(
            required_native_fragment in name for name in native_names
        ):
            return None

        digest = hashlib.sha256()
        digest.update(b"exo-sglang-kt-artifact-v1\0")
        for relative_name, artifact_path in sorted(artifact_files):
            encoded_name = relative_name.encode("utf-8")
            digest.update(len(encoded_name).to_bytes(8, "big"))
            digest.update(encoded_name)
            with artifact_path.open("rb") as artifact_file:
                while chunk := artifact_file.read(1024 * 1024):
                    digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None
""".strip()
