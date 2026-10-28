"""Sistema de notificaciones - version DEMO/simulacion.

Este modulo NO envia mensajes reales. Prepara la arquitectura para que en una
fase posterior se pueda integrar un canal de notificaciones (correo, SMS,
webhook, etc.) mediante configuracion externa segura.

Para habilitar notificaciones reales en el futuro se recomienda:
- Leer destinatarios y credenciales desde variables de entorno o un gestor de secretos.
- Nunca almacenar credenciales en el codigo fuente.
- Implementar un backend de cola o tareas asincronas para evitar bloquear la UI.
- Validar el destinatario y obtener autorizacion explicita antes de enviar alertas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from app_modules.alert_manager import MonitoredEvent


@dataclass
class NotificationLog:
    """Registro de una notificacion simulada."""

    notification_id: str
    event_id: str
    event_type: str
    recipient_type: str
    channel: str
    simulated_at: datetime
    message_preview: str
    sent: bool = False
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ID notificacion": self.notification_id,
            "ID evento": self.event_id,
            "Tipo evento": self.event_type,
            "Destinatario": self.recipient_type,
            "Canal": self.channel,
            "Simulado": self.simulated_at.strftime("%Y-%m-%d %H:%M:%S"),
            "Mensaje": self.message_preview,
            "Enviado real": self.sent,
            "Nota": self.note,
        }


NOTIFICATION_COUNTER = 0


def _next_notification_id() -> str:
    global NOTIFICATION_COUNTER
    NOTIFICATION_COUNTER += 1
    return f"NOTIF-{NOTIFICATION_COUNTER:04d}"


def _build_message_preview(event: MonitoredEvent) -> str:
    return (
        f"[DEMO] Alerta {event.event_type} (prioridad {event.priority}) "
        f"detectada por {', '.join(event.supporting_models)}. "
        f"Confianza: {event.confidence:.2f}. Requiere verificacion."
    )


def simulate_notification(
    event: MonitoredEvent,
    recipient_type: str = "Operador de sala de control",
    channel: str = "Dashboard + correo (pendiente de configuracion)",
    logs: Optional[List[NotificationLog]] = None,
) -> NotificationLog:
    """Simula el envio de una notificacion para un evento."""
    log = NotificationLog(
        notification_id=_next_notification_id(),
        event_id=event.event_id,
        event_type=event.event_type,
        recipient_type=recipient_type,
        channel=channel,
        simulated_at=datetime.now(),
        message_preview=_build_message_preview(event),
        sent=False,
        note=(
            "Modo demostracion: no se ha enviado ningun mensaje real. "
            "Configure un canal de notificaciones autorizado para habilitar envios."
        ),
    )
    if logs is not None:
        logs.append(log)
    return log


def get_notification_config_placeholder() -> Dict[str, Any]:
    """Devuelve un esquema de configuracion futuro."""
    return {
        "enabled": False,
        "channels": {
            "email": {
                "smtp_server": "smtp.ejemplo.com",
                "port": 587,
                "username": "${EMAIL_USER}",
                "password": "${EMAIL_PASS}",
                "recipients": ["operador@ejemplo.com"],
            },
            "sms": {
                "provider": "pendiente",
                "phone_numbers": [],
            },
            "webhook": {
                "url": "https://ejemplo.com/webhook",
                "headers": {"Authorization": "Bearer ${WEBHOOK_TOKEN}"},
            },
        },
        "note": "Esta configuracion es un placeholder. No se usan credenciales reales.",
    }
