"""Research tools for computational temporal-activity reduction in video."""

from .analyzers import AnalyzerConfig
from .api import AnalysisReport, FilterReport, analyze_video, filter_video
from .filtering import AdaptiveFilterConfig

__all__ = [
    "AdaptiveFilterConfig",
    "AnalysisReport",
    "AnalyzerConfig",
    "FilterReport",
    "analyze_video",
    "filter_video",
]

__version__ = "0.1.0"
