from typing import Tuple, List, Literal
_UNSET = object()

# Universal types
class Universal:
    SearchFolderOrListFiles = Tuple[str, str] | List[str]
    CreateInFolderOrListFiles = Tuple[str, str] | List[str]
    SaveAsCog= bool # Default: True
    DebugLogs= bool # Default: False

    @staticmethod
    def validate(
        *,
        input_images=_UNSET,
        output_images=_UNSET,
        save_as_cog=_UNSET,
        debug_logs=_UNSET,
        ):
        if input_images is not _UNSET:
            if not isinstance(input_images, (tuple, list)):
                raise ValueError("input_images must be a tuple (folder, pattern) or a list of strings.")
            if isinstance(input_images, tuple):
                if len(input_images) != 2 or not all(isinstance(s, str) for s in input_images):
                    raise ValueError("If input_images is a tuple, it must be (folder_path, pattern).")
            elif not all(isinstance(p, str) for p in input_images):
                raise ValueError("All elements in input_images list must be strings.")

        if output_images is not _UNSET:
            if not isinstance(output_images, (tuple, list)):
                raise ValueError("output_images must be a tuple or a list of strings.")
            if isinstance(output_images, tuple):
                if len(output_images) != 2 or not all(isinstance(s, str) for s in output_images):
                    raise ValueError("If output_images is a tuple, it must be (output_folder, name_template).")
            elif not all(isinstance(p, str) for p in output_images):
                raise ValueError("All elements in output_images list must be strings.")

        if save_as_cog is not _UNSET:
            if not isinstance(save_as_cog, bool):
                raise ValueError("save_as_cog must be a boolean.")

        if debug_logs is not _UNSET:
            if not isinstance(debug_logs, bool):
                raise ValueError("debug_logs must be a boolean.")


# Match types
class Match:
    VectorMask = Tuple[Literal["include", "exclude"], str] | Tuple[Literal["include", "exclude"], str, str] | None # Default: None
    WindowSize = int | Tuple[int, int] | Literal["internal"] | None # Default: None
    WindowSizeWihBlock = int | Tuple[int, int] | Literal["internal"] | Literal["block"] | None # Default: None
    CustomNodataValue = float | int | None # Default: None
    ImageParallelWorkers = Tuple[Literal["process"], Literal["cpu"] | int] | None # Default: None
    WindowParallelWorkers = Tuple[Literal["process"], Literal["cpu"] | int] | None # Default: None
    CalculationDtype = str # Default: "float32"
    OutputDtype = str | None # Default: None
    SpecifyModelImages = Tuple[Literal["exclude", "include"], List[str]] | None # Default: None

    @staticmethod
    def validate_match(
        *,
        vector_mask=_UNSET,
        window_size=_UNSET,
        custom_nodata_value=_UNSET,
        image_parallel_workers=_UNSET,
        window_parallel_workers=_UNSET,
        calculation_dtype=_UNSET,
        output_dtype=_UNSET,
        specify_model_images=_UNSET,
        ):
        if vector_mask is not _UNSET and vector_mask is not None:
            if not isinstance(vector_mask, tuple) or len(vector_mask) not in {2, 3}:
                raise ValueError("vector_mask must be a tuple of 2 or 3 elements.")
            if vector_mask[0] not in {"include", "exclude"}:
                raise ValueError("The first element of vector_mask must be 'include' or 'exclude'.")
            if not isinstance(vector_mask[1], str):
                raise ValueError("The second element must be a string (vector file path).")
            if len(vector_mask) == 3 and not isinstance(vector_mask[2], str):
                raise ValueError("The third element, if provided, must be a string (field name).")

        if window_size is not _UNSET:
            def _validate_window_param(val):
                if val is None or isinstance(val, int):
                    return
                if isinstance(val, tuple) and len(val) == 2 and all(isinstance(i, int) for i in val):
                    return
                if val == "internal":
                    return
                raise ValueError("window_size must be an int, (w, h) tuple, 'internal', or None.")
            _validate_window_param(window_size)

        if custom_nodata_value is not _UNSET:
            if custom_nodata_value is not None and not isinstance(custom_nodata_value, (int, float)):
                raise ValueError("custom_nodata_value must be a number or None.")

        def _validate_parallel_workers(val, name):
            if val is None:
                return
            if not isinstance(val, tuple) or len(val) != 2:
                raise ValueError(f"{name} must be a tuple of (backend, workers) or None.")
            backend, workers = val
            if backend != "process":
                raise ValueError(f"The first element of {name} must be 'process'.")
            if workers != "cpu" and not isinstance(workers, int):
                raise ValueError(f"The second element of {name} must be 'cpu' or an integer.")

        if image_parallel_workers is not _UNSET:
            _validate_parallel_workers(image_parallel_workers, "image_parallel_workers")

        if window_parallel_workers is not _UNSET:
            _validate_parallel_workers(window_parallel_workers, "window_parallel_workers")

        if calculation_dtype is not _UNSET:
            if not isinstance(calculation_dtype, str):
                raise ValueError("calculation_dtype must be a string.")

        if output_dtype is not _UNSET and output_dtype is not None:
            if not isinstance(output_dtype, str):
                raise ValueError("output_dtype must be a string or None.")

        if specify_model_images is not _UNSET and specify_model_images is not None:
            if (
                not isinstance(specify_model_images, tuple)
                or len(specify_model_images) != 2
                or specify_model_images[0] not in {"include", "exclude"}
                or not isinstance(specify_model_images[1], list)
                or not all(isinstance(s, str) for s in specify_model_images[1])
            ):
                raise ValueError("specify_model_images must be a tuple of ('include'|'exclude', list of strings).")

    @staticmethod
    def validate_global_regression(
        *,
        custom_mean_factor=_UNSET,
        custom_std_factor=_UNSET,
        save_adjustments=_UNSET,
        load_adjustments=_UNSET,
        ):
        if custom_mean_factor is not _UNSET:
            if not isinstance(custom_mean_factor, (int, float)):
                raise ValueError("custom_mean_factor must be a number.")

        if custom_std_factor is not _UNSET:
            if not isinstance(custom_std_factor, (int, float)):
                raise ValueError("custom_std_factor must be a number.")

        if save_adjustments is not _UNSET and save_adjustments is not None:
            if not isinstance(save_adjustments, str):
                raise ValueError("save_adjustments must be a string or None.")

        if load_adjustments is not _UNSET and load_adjustments is not None:
            if not isinstance(load_adjustments, str):
                raise ValueError("load_adjustments must be a string or None.")