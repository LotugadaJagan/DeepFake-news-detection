from app import app

# This file is used by Gunicorn to serve the app
# The app should be already initialized in app.py

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
