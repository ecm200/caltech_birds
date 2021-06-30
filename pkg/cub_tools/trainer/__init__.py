from .Base import Trainer
from .Ignite import Ignite_Trainer

try:
    import clearml
except:
    print('[WARNING] CLEARML package was not detected, therefore CLEARML Trainer class will not be imported.')
    __all__ = ["Trainer", "Ignite_Trainer"]
else:
    from .IgniteClearML import ClearML_Ignite_Trainer
    __all__ = ["Trainer", "Ignite_Trainer", "ClearML_Ignite_Trainer"]