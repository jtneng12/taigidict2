# Not meant to be used as-is - PythonAnywhere auto-generates its own WSGI
# config file (its path is shown on your Web app's "Web" tab, something like
# /var/www/<username>_pythonanywhere_com_wsgi.py) with its own boilerplate.
# Open THEIR generated file and add these lines near the bottom, replacing
# whatever placeholder "application" line it ships with.
#
# Change <username> and the project folder name below to match wherever you
# actually uploaded this bundle.

import sys

project_home = '/home/<username>/taigidict'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# gemini_app.py's own app.run(...) line at the bottom never executes here -
# PythonAnywhere's WSGI server calls `application` directly, it doesn't run
# this file as __main__. No code changes were needed in gemini_app.py itself.
from gemini_app import app as application
