"""Celery worker package for DominicBE.

Keep this package lightweight: importing ``app.worker`` must not create the
FastAPI application or import API endpoint modules.
"""

__all__: list[str] = []
