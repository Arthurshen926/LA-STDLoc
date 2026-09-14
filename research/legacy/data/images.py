"""Image-resolution conventions shared by cache and frontend code."""


def resolution_from_longest_edge(height: int, width: int, longest_edge: int = 640):
    height, width = int(height), int(width)
    longest_edge = int(longest_edge or 0)
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if longest_edge <= 0:
        return height, width
    if height > width:
        return longest_edge, int(width * longest_edge / height)
    return int(height * longest_edge / width), longest_edge
