"""
Handlers to create the Graphical User Interfaces.
"""

import PySimpleGUI as sg

# Create a custom theme
sg.LOOK_AND_FEEL_TABLE['VSCodeDark'] = {
    'BACKGROUND': '#1e1e1e',        # VS Code editor background
    'TEXT': '#ffffff',              # White text (default)
    'INPUT': '#252526',             # Dark gray input fields
    'TEXT_INPUT': '#ffffff',
    'SCROLL': '#333333',
    'BUTTON': ('#ffffff', '#0e639c'),  # White text on VS Code blue
    'PROGRESS': ('#ffffff', '#0e639c'),
    'BORDER': 1,
    'SLIDER_DEPTH': 0,
    'PROGRESS_DEPTH': 0,
}