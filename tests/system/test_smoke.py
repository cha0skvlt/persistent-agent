import os
import sys


def test_build_app_memory_checkpointer():
    os.environ["CHECKPOINTER"] = "memory"
    sys.path.insert(0, ".")

    from src.config import settings

    settings.checkpointer = "memory"

    from src.graph.workflow import build_app

    app = build_app()
    assert app is not None
