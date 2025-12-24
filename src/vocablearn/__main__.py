"""Allow running vocablearn as a module: python -m vocablearn"""

from vocablearn.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
