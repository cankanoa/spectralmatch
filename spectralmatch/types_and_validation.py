from typing import Tuple, List, Literal, Optional

_UNSET = object()


# Universal types
class Universal:
    SearchFolderOrListFiles = str | List[str]
    CreateInFolderOrListFiles = str | List[str]
    SaveAsCog = bool  # Default: True
    DebugLogs = bool  # Default: False
    VectorMask = Tuple[Literal["include", "exclude"], str, Optional[str]] | None
    WindowSize = int | Tuple[int, int] | Literal["internal"] | None
    WindowSizeWithBlock = int | Tuple[int, int] | Literal["internal", "block"] | None
    CustomNodataValue = float | int | None
    ImageParallelWorkers = (
        Tuple[Literal["process", "thread"], Literal["cpu"] | int] | None
    )
    WindowParallelWorkers = Tuple[Literal["process"], Literal["cpu"] | int] | None
    CalculationDtype = str
    CustomOutputDtype = str | None
    CreateNameAttribute: Tuple[str, str] | None

    @staticmethod
    def validate(
        *,
        input_images=_UNSET,
        output_images=_UNSET,
        save_as_cog=_UNSET,
        debug_logs=_UNSET,
        vector_mask=_UNSET,
        window_size=_UNSET,
        custom_nodata_value=_UNSET,
        image_parallel_workers=_UNSET,
        window_parallel_workers=_UNSET,
        calculation_dtype=_UNSET,
        custom_output_dtype=_UNSET,
        create_name_attribute=_UNSET,
        output_dtype=_UNSET,
    ):
        if input_images is not _UNSET:
            if not isinstance(input_images, (str, list)):
                raise ValueError(
                    "input_images must be a string (path or glob pattern) or a list of strings."
                )
            if isinstance(input_images, list) and not all(
                isinstance(p, str) for p in input_images
            ):
                raise ValueError("All elements in input_images list must be strings.")

        if output_images is not _UNSET:
            if not isinstance(output_images, (str, list)):
                raise ValueError(
                    "output_images must be a string (path or template) or a list of strings."
                )
            if isinstance(output_images, list) and not all(
                isinstance(p, str) for p in output_images
            ):
                raise ValueError("All elements in output_images list must be strings.")

        if save_as_cog is not _UNSET:
            if not isinstance(save_as_cog, bool):
                raise ValueError("save_as_cog must be a boolean.")

            if save_as_cog:
                if window_size is _UNSET or window_size is None:
                    raise ValueError("When save_as_cog=True, window_size must be set.")
                if isinstance(window_size, int):
                    if window_size % 16 != 0:
                        raise ValueError(
                            "When save_as_cog=True, window_size must be a multiple of 16."
                        )
                elif isinstance(window_size, tuple):
                    if len(window_size) != 2 or window_size[0] != window_size[1]:
                        raise ValueError(
                            "When save_as_cog=True, window_size must be square (width == height)."
                        )
                    if any(w % 16 != 0 for w in window_size):
                        raise ValueError(
                            "When save_as_cog=True, window_size dimensions must be multiples of 16."
                        )

        if debug_logs is not _UNSET:
            if not isinstance(debug_logs, bool):
                raise ValueError("debug_logs must be a boolean.")

        if vector_mask is not _UNSET and vector_mask is not None:
            if not isinstance(vector_mask, tuple) or len(vector_mask) not in {2, 3}:
                raise ValueError("vector_mask must be a tuple of 2 or 3 elements.")
            if vector_mask[0] not in {"include", "exclude"}:
                raise ValueError(
                    "The first element of vector_mask must be 'include' or 'exclude'."
                )
            if not isinstance(vector_mask[1], str):
                raise ValueError(
                    "The second element must be a string (vector file path)."
                )
            if len(vector_mask) == 3 and not isinstance(vector_mask[2], str):
                raise ValueError(
                    "The third element, if provided, must be a string (field name)."
                )

        if window_size is not _UNSET:

            def _validate_window_param(val):
                if val is None or isinstance(val, int):
                    return
                if (
                    isinstance(val, tuple)
                    and len(val) == 2
                    and all(isinstance(i, int) for i in val)
                ):
                    return
                if val == "internal":
                    return
                raise ValueError(
                    "window_size must be an int, (w, h) tuple, 'internal', or None."
                )

            _validate_window_param(window_size)

        if custom_nodata_value is not _UNSET:
            if custom_nodata_value is not None and not isinstance(
                custom_nodata_value, (int, float)
            ):
                raise ValueError("custom_nodata_value must be a number or None.")

        if image_parallel_workers is not _UNSET:
            if image_parallel_workers is not None:
                if (
                    not isinstance(image_parallel_workers, tuple)
                    or len(image_parallel_workers) != 2
                    or image_parallel_workers[0] not in {"process", "thread"}
                    or (
                        image_parallel_workers[1] != "cpu"
                        and not isinstance(image_parallel_workers[1], int)
                    )
                ):
                    raise ValueError(
                        "image_parallel_workers must be a tuple like ('process'|'thread', 'cpu'|int) or None."
                    )

        if window_parallel_workers is not _UNSET:
            if window_parallel_workers is not None:
                if (
                    not isinstance(window_parallel_workers, tuple)
                    or len(window_parallel_workers) != 2
                    or window_parallel_workers[0] != "process"
                    or (
                        window_parallel_workers[1] != "cpu"
                        and not isinstance(window_parallel_workers[1], int)
                    )
                ):
                    raise ValueError(
                        "window_parallel_workers must be a tuple like ('process', 'cpu'|int) or None."
                    )

        if calculation_dtype is not _UNSET:
            if not isinstance(calculation_dtype, str):
                raise ValueError("calculation_dtype must be a string.")

        if custom_output_dtype is not _UNSET and custom_output_dtype is not None:
            if not isinstance(custom_output_dtype, str):
                raise ValueError("custom_output_dtype must be a string or None.")

        if create_name_attribute is not _UNSET:
            if (
                not isinstance(Universal.CreateNameAttribute, tuple)
                or len(Universal.CreateNameAttribute) != 2
            ):
                raise ValueError(
                    "CreateNameAttribute must be a tuple of two strings or None."
                )
            if not all(isinstance(s, str) for s in Universal.CreateNameAttribute):
                raise ValueError(
                    "Both elements of CreateNameAttribute must be strings."
                )
        if output_dtype is not _UNSET and output_dtype is not None:
            if not isinstance(output_dtype, str):
                raise ValueError("output_dtype must be a string or None.")


# Match-specific only
class Match:
    SpecifyModelImages = Tuple[Literal["exclude", "include"], List[str]] | None

    @staticmethod
    def validate_match(
        *,
        specify_model_images=_UNSET,
    ):
        if specify_model_images is not _UNSET and specify_model_images is not None:
            if (
                not isinstance(specify_model_images, tuple)
                or len(specify_model_images) != 2
                or specify_model_images[0] not in {"include", "exclude"}
                or not isinstance(specify_model_images[1], list)
                or not all(isinstance(s, str) for s in specify_model_images[1])
            ):
                raise ValueError(
                    "specify_model_images must be a tuple of ('include'|'exclude', list of strings)."
                )

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

    @staticmethod
    def validate_local_block_adjustment(
        *,
        number_of_blocks=_UNSET,
        alpha=_UNSET,
        correction_method=_UNSET,
        save_block_maps=_UNSET,
        load_block_maps=_UNSET,
        override_bounds_canvas_coords=_UNSET,
        block_valid_pixel_threshold=_UNSET,
    ):
        if number_of_blocks is not _UNSET:
            if not (
                isinstance(number_of_blocks, int)
                or (
                    isinstance(number_of_blocks, tuple)
                    and len(number_of_blocks) == 2
                    and all(isinstance(i, int) for i in number_of_blocks)
                )
                or number_of_blocks == "coefficient_of_variation"
            ):
                raise ValueError(
                    "number_of_blocks must be an int, a (width, height) tuple, or 'coefficient_of_variation'."
                )

        if alpha is not _UNSET:
            if not isinstance(alpha, (float, int)):
                raise ValueError("alpha must be a float or int.")

        if correction_method is not _UNSET:
            if correction_method not in {"gamma", "linear"}:
                raise ValueError(
                    "correction_method must be either 'gamma' or 'linear'."
                )

        if save_block_maps is not _UNSET:
            if save_block_maps is not None:
                if not (
                    isinstance(save_block_maps, tuple)
                    and len(save_block_maps) == 2
                    and all(isinstance(s, str) for s in save_block_maps)
                ):
                    raise ValueError(
                        "save_block_maps must be a tuple of two strings or None."
                    )

        if load_block_maps is not _UNSET:
            if load_block_maps is not None:
                if not (
                    isinstance(load_block_maps, tuple)
                    and len(load_block_maps) == 2
                    and (
                        (
                            isinstance(load_block_maps[0], str)
                            or load_block_maps[0] is None
                        )
                        and (
                            isinstance(load_block_maps[1], list)
                            or load_block_maps[1] is None
                        )
                    )
                ):
                    raise ValueError(
                        "load_block_maps must be a tuple (str|None, list[str]|None) or None."
                    )

        if override_bounds_canvas_coords is not _UNSET:
            if override_bounds_canvas_coords is not None:
                if not (
                    isinstance(override_bounds_canvas_coords, tuple)
                    and len(override_bounds_canvas_coords) == 4
                    and all(
                        isinstance(v, (float, int))
                        for v in override_bounds_canvas_coords
                    )
                ):
                    raise ValueError(
                        "override_bounds_canvas_coords must be a tuple of four floats or ints, or None."
                    )

        if block_valid_pixel_threshold is not _UNSET:
            if not isinstance(block_valid_pixel_threshold, (float, int)) or not (
                0 <= block_valid_pixel_threshold <= 1
            ):
                raise ValueError(
                    "block_valid_pixel_threshold must be a float between 0 and 1."
                )
