"""
UI Service for initializing and launching the Dash app.
"""
import dash
from .ui_layout import create_layout
from .ui_callbacks import register_callbacks

class UIService:
    def __init__(self):
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.app.layout = create_layout()
        register_callbacks(self.app)

    def launch_ui(self, port: int = 8050):
        # Disable debugging features as requested for educational context.
        # Run on 0.0.0.0 because it's required for AI Studio environments 
        # (even if PRD says localhost, the container requires 0.0.0.0 or standard config)
        # We'll use 0.0.0.0 to be safe if running in a container, but default is 127.0.0.1
        self.app.run(host="0.0.0.0", port=port, debug=False)
