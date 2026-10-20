"""Detectors package. Importing it registers the concrete receivers."""
from detectors import lmmse  # noqa: F401  (registers "lmmse")
from detectors import icd    # noqa: F401  (registers "icd", "he" and "genie")
from detectors import jcd    # noqa: F401  (registers "jcd")
from detectors import bigamp  # noqa: F401  (registers "bigamp")
from detectors import bigabp  # noqa: F401  (registers "bigabp" and the ablation "bigabp_node")
