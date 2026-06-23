import math

class Grid:
    def __init__(self, x_min, x_max, y_min, y_max, row, column):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.row = row
        self.column = column
        self.width = (self.x_max - self.x_min) / self.column
        self.height = (self.y_max - self.y_min) / self.row

    def calculate_grid_coordinates(self, x_pos, y_pos):
        # reject out-of-arena / invalid values
        if x_pos is None or y_pos is None:
            return None

        if x_pos < self.x_min or x_pos >= self.x_max:
            return None
        if y_pos < self.y_min or y_pos >= self.y_max:
            return None

        col = math.floor((x_pos - self.x_min) / self.width)
        row = math.floor((y_pos - self.y_min) / self.height)

        # safety clamp
        col = max(0, min(col, self.column - 1))
        row = max(0, min(row, self.row - 1))

        return (row, col)