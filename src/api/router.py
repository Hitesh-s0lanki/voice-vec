"""Aggregates every controller under a single versioned router."""

from fastapi import APIRouter

from src.controllers import (
    ask_controller,
    health_controller,
    metrics_controller,
    suggestions_controller,
    voice_controller,
)

api_router = APIRouter()
api_router.include_router(health_controller.router)
api_router.include_router(ask_controller.router)
api_router.include_router(metrics_controller.router)
api_router.include_router(suggestions_controller.router)
api_router.include_router(voice_controller.router)
