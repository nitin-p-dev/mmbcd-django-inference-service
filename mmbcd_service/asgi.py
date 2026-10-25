"""
mmbcd_service/asgi.py
=====================
ASGI config for mmbcd_service project.
"""

import os
from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mmbcd_service.settings")

application = get_asgi_application()
