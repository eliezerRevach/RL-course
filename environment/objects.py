from minigrid.core.world_object import WorldObj, Box

class SmallBox(Box):
    """
    A small box that can be pushed by a single agent.
    """
    def __init__(self, color="yellow"):
        super().__init__(color)
        # Custom property to differentiate from regular minigrid boxes
        self.box_size = "small"

    def can_overlap(self):
        """Objects can't overlap with the box"""
        return False
        
    def can_pickup(self):
        """Boxes cannot be picked up, only pushed"""
        return False

class HeavyBox(Box):
    """
    A heavy box that requires two agents pushing from the exact same grid space simultaneously.
    """
    def __init__(self, color="purple"):
        super().__init__(color)
        self.box_size = "heavy"

    def can_overlap(self):
        """Objects can't overlap with the box"""
        return False
        
    def can_pickup(self):
        """Boxes cannot be picked up, only pushed"""
        return False
