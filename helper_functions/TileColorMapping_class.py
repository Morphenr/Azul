class TileColorMapping:
    def __init__(self, tile_colors):
        """
        Initialize the tile color mapping using the list of tile colors from the game settings.
        Each color will be mapped to an integer starting from 0.
        """
        self.tile_colors = tile_colors
        self.mapping = {color: idx for idx, color in enumerate(tile_colors)}
    
    def get(self, color, default=-1):
        """
        Get the numeric value of a tile color. If the color is not found, return the default value.
        """
        return self.mapping.get(color, default)

    def __repr__(self):
        return f"TileColorMapping({self.mapping})"
