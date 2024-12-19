class ThemeManager:
    @staticmethod
    def adjust_color_brightness(color_hex, factor):
        """Adjust the brightness of a hex color"""
        # Convert hex to RGB
        color = color_hex.lstrip('#')
        rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        
        # Adjust brightness
        new_rgb = tuple(min(255, int(c * factor)) for c in rgb)
        
        # Convert back to hex
        return '#{:02x}{:02x}{:02x}'.format(*new_rgb)

    @staticmethod
    def get_color_schemes():
        """Get default color schemes"""
        return {
            'light': {
                'bg': '#ffffff',
                'fg': '#2c2c2c',
                'button_bg': {
                    'primary': '#2196F3',    # Blue
                    'success': '#4CAF50',    # Green
                    'warning': '#FF9800',    # Orange
                    'danger': '#f44336',     # Red
                    'purple': '#9C27B0',     # Purple
                    'indigo': '#673AB7'      # Indigo
                },
                'button_fg': '#ffffff',
                'frame_bg': '#f5f5f5',
                'input_bg': '#ffffff',
                'input_fg': '#2c2c2c'
            },
            'dark': {
                'bg': '#1e1e1e',
                'fg': '#ffffff',
                'button_bg': {
                    'primary': '#1976D2',    # Darker Blue
                    'success': '#388E3C',    # Darker Green
                    'warning': '#F57C00',    # Darker Orange
                    'danger': '#D32F2F',     # Darker Red
                    'purple': '#7B1FA2',     # Darker Purple
                    'indigo': '#512DA8'      # Darker Indigo
                },
                'button_fg': '#ffffff',
                'frame_bg': '#2d2d2d',
                'input_bg': '#2d2d2d',
                'input_fg': '#ffffff'
            }
        } 