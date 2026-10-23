"""Development entry point.

    python app.py                       # http://127.0.0.1:5000
    gunicorn "app:app"                  # production (see Dockerfile)
"""

import os

from ocular import create_app

app = create_app()

if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
