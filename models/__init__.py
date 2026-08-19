# 智慧养蜂蜜蜂识别与行为量化研究项目

from .annotator import (
    BeeAnnotationConfig,
    VideoAnnotationExtractor,
    COCOAnnotationConverter,
    MOTAnnotationConverter,
    ManualAnnotationTool,
    YOLOAnnotationConverter,
    create_sample_annotations
)

from .outside_tracker import (
    OutsideHiveBeeDetector,
    DeepSORTTracker,
    ByteTracker,
    OutsideHiveTracker,
    create_outside_tracker
)

from .inside_tracker import (
    InfraredImageEnhancer,
    BeePoseEstimator,
    InsideHiveBeeDetector,
    InsideHiveTracker,
    InsideHiveAnalyzer,
    create_inside_tracker
)

from .quantifier import (
    IndividualBehavior,
    GroupBehavior,
    TrackletBuilder,
    BehaviorClassifier,
    ActivityIntensityAnalyzer,
    SpatialDensityAnalyzer,
    BehaviorQuantifier,
    HiveEntranceAnalyzer,
    create_behavior_quantifier
)

from .visualizer import (
    ColorGenerator,
    TrackVisualizer,
    BehaviorVisualizer,
    DensityMapVisualizer,
    StatisticsPlotter,
    VideoAnnotator,
    create_visualizer
)

from .processor import (
    OutsideHiveProcessor,
    InsideHiveProcessor,
    MultiModalProcessor,
    load_config
)

__version__ = '1.0.0'
