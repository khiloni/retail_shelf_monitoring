"""Custom exceptions for the shelf_monitor package."""


class ShelfMonitorError(Exception):
    """Base exception."""


class ConfigurationError(ShelfMonitorError):
    """Bad configuration."""


class ModelNotFoundError(ShelfMonitorError):
    """Required model file is missing."""


class ModelCorruptError(ShelfMonitorError):
    """Model file exists but fails to load or validate."""


class InferenceError(ShelfMonitorError):
    """Error during model inference."""


class ValidationError(ShelfMonitorError):
    """Data validation failure."""


class AlignmentError(ShelfMonitorError):
    """Shelf alignment failure."""


class PlanogramNotFoundError(ShelfMonitorError):
    """No planogram found for the requested shelf."""
