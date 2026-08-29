"""Aggregates every controller under a single versioned router."""

from fastapi import APIRouter

from src.controllers import (
    ask_controller,
    connectors_controller,
    conversations_controller,
    datasets_controller,
    health_controller,
    integrations_controller,
    metrics_controller,
    voice_controller,
)

api_router = APIRouter()
api_router.include_router(health_controller.router)
api_router.include_router(connectors_controller.router)
api_router.include_router(integrations_controller.router)
api_router.include_router(ask_controller.router)
api_router.include_router(conversations_controller.router)
api_router.include_router(datasets_controller.router)
api_router.include_router(metrics_controller.router)
api_router.include_router(voice_controller.router)
