"""OpenTelemetry bootstrap for Agent. Idempotent; safe to import repeatedly.

Uses OTLP/HTTP (not gRPC). Heart and other daemons ``subprocess``-fork; gRPC's
``cygrpc`` threads make macOS abort the child with
``multi-threaded process forked`` / SIGTRAP (see DiagnosticReports).
"""

from __future__ import annotations

import logging
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_log = logging.getLogger(__name__)

_initialized = False
_degraded = False
_httpx_instrumented = False
_log_handler: LoggingHandler | None = None
# Providers installed by the last ``init_telemetry`` — kept only so tests can
# shut their exporter threads down again (see ``reset_telemetry_for_tests``).
_providers: list[object] = []


def is_telemetry_enabled() -> bool:
    """True after ``init_telemetry`` registered providers for this process."""
    return _initialized and not _degraded


def _collector_reachable(endpoint: str, *, timeout: float = 1.0) -> bool:
    """Fast TCP probe of the OTLP collector before installing exporters."""
    import socket
    from urllib.parse import urlparse

    try:
        parsed = urlparse(endpoint)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 4318)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
    except Exception:
        return True  # unparsable endpoint — let the exporter surface it


def reset_telemetry_for_tests() -> None:
    """Clear init flag and shut down any exporters this process installed.

    Shutting down matters: ``PeriodicExportingMetricReader`` runs a background
    thread that keeps POSTing on its own interval. Left alive, it outlives the
    test that created it and floods the rest of the run with connection errors
    against an endpoint no collector is listening on.
    """
    global _initialized, _degraded, _httpx_instrumented, _log_handler, _providers
    if _log_handler is not None:
        for name in ("agent.event", "src.heart.heartbeat"):
            logging.getLogger(name).removeHandler(_log_handler)
    for provider in _providers:
        shutdown = getattr(provider, "shutdown", None)
        if shutdown is None:
            continue
        try:
            shutdown()
        except Exception:  # a dead exporter must not break the reset
            pass
    _providers = []
    from src.obs.metrics import _reset_instruments_for_tests

    _reset_instruments_for_tests()
    _initialized = False
    _degraded = False
    _httpx_instrumented = False
    _log_handler = None


def _otlp_http_endpoint(raw: str) -> str:
    """Normalize settings endpoint to an OTLP/HTTP base URL.

    Accepts ``host:port``, ``http(s)://…``. Legacy gRPC port ``4317`` maps to
    collector HTTP ``4318`` (optional external collector; none shipped in-repo).
    """
    ep = (raw or "127.0.0.1:4318").strip()
    if not ep:
        ep = "127.0.0.1:4318"

    if ep.startswith("http://") or ep.startswith("https://"):
        # http://127.0.0.1:4317 → :4318 (old gRPC default left in .env)
        if ep.rstrip("/").endswith(":4317"):
            ep = ep.rstrip("/")[:-4] + "4318"
        return ep.rstrip("/")

    hostport = ep
    if hostport.endswith(":4317"):
        hostport = hostport[:-4] + "4318"
    elif ":" not in hostport:
        hostport = f"{hostport}:4318"
    return f"http://{hostport}"


def init_telemetry(service_name: str, settings: Any) -> None:
    """Bootstrap OTel providers + auto-instrumentation. Call once at process start."""
    global _initialized, _httpx_instrumented, _log_handler, _providers
    if _initialized or not bool(getattr(settings, "otel_enabled", False)):
        return

    endpoint = _otlp_http_endpoint(
        getattr(settings, "otel_endpoint", None) or "127.0.0.1:4318"
    )

    # Collector down → degrade silently: one WARNING instead of an endless
    # "Exception while exporting" retry storm every batch interval.
    if not _collector_reachable(endpoint):
        global _degraded
        _log.warning(
            "otel collector unreachable at %s — telemetry disabled this run "
            "(set OTEL_ENABLED=0 or provide a reachable collector)",
            endpoint,
        )
        _degraded = True
        _initialized = True
        return

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.namespace": "agent",
            "deployment.environment": str(
                getattr(settings, "environment", "dev") or "dev"
            ),
        }
    )

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=endpoint),
        )
    )
    trace.set_tracer_provider(tracer_provider)

    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[
            PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=endpoint),
                export_interval_millis=15_000,
            )
        ],
    )
    metrics.set_meter_provider(meter_provider)
    from src.obs.metrics import _bind_instruments

    _bind_instruments()

    log_provider = LoggerProvider(resource=resource)
    log_provider.add_log_record_processor(
        BatchLogRecordProcessor(
            OTLPLogExporter(endpoint=endpoint),
        )
    )
    set_logger_provider(log_provider)
    _providers = [tracer_provider, meter_provider, log_provider]
    _log_handler = LoggingHandler(level=logging.NOTSET, logger_provider=log_provider)
    for log_name in ("agent.event", "src.heart.heartbeat"):
        event_log = logging.getLogger(log_name)
        # Keep propagation on: root file/console handlers must still receive
        # these events, or a downed collector loses them entirely. Root has no
        # OTLP handler, so this does not duplicate records in OTLP.
        event_log.propagate = True
        if _log_handler not in event_log.handlers:
            event_log.addHandler(_log_handler)

    if not _httpx_instrumented:
        HTTPXClientInstrumentor().instrument()
        _httpx_instrumented = True
    LoggingInstrumentor().instrument(set_logging_format=True)

    _initialized = True
    _log.info(
        "OpenTelemetry enabled service=%s endpoint=%s protocol=http",
        service_name,
        endpoint,
    )


def get_tracer(name: str = "agent") -> trace.Tracer:
    return trace.get_tracer(name)


def get_meter(name: str = "agent") -> metrics.Meter:
    return metrics.get_meter(name)


__all__ = [
    "get_meter",
    "get_tracer",
    "init_telemetry",
    "is_telemetry_enabled",
    "reset_telemetry_for_tests",
]
