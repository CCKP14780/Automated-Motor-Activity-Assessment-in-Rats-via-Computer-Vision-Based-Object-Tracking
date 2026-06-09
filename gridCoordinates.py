import math
import json

with open('data.json', 'r') as f:
    config = json.load(f)

class Grid():
    def __init__(self, x_min, x_max, y_min, y_max, row, column):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.row = row
        self.column = column
        self.width = (self.x_max - self.x_min) / self.column
        self.height = (self.y_max - self.y_min) / self.row
        self.body_parts = config.get('BODY_PARTS')
        
    def grid_coords_list(self):
        grid_coordinates = {}
        for r in range(self.row):
            for c in range(self.column):
                grid_coordinates[(r, c)] = 0
        return grid_coordinates
    
    def calculate_grid_coordinates(self, x_pos, y_pos, body_part):
        col = math.floor((x_pos - self.x_min) / self.width)
        row = math.floor((y_pos - self.y_min) / self.height)

        if (row, col) in self.grid_coords_list():
            self.grid_coords_list()[(row, col)] += 1
        return (row, col)
