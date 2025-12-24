"""Storage backends for vocablearn.

Provides database abstraction for vocabulary storage.
"""

from vocablearn.storage.sqlite import SQLiteStorage

__all__ = ["SQLiteStorage"]
