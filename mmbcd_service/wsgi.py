"""
mmbcd_service/wsgi.py
=====================
WSGI config for mmbcd_service project.
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mmbcd_service.settings")

application = get_wsgi_application()
