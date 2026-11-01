# Model and heuristic thresholds live here, not in env: they are tuned together with the code


class ENGINE_SETTINGS:
    """Inference engine shared by the services."""
    WARMUP_CALLS = 3                    # idle calls at startup: kernels, allocators and caches are ready before the first request
