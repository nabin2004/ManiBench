from manim import *

class CollidingBlocksScene(Scene):
    def construct(self):
        title = Title('Colliding Blocks & Pi')
        self.play(Write(title))
        wall = Line(LEFT * 5 + DOWN * 2, LEFT * 5 + UP * 2)
        floor = Line(LEFT * 6 + DOWN * 2, RIGHT * 6 + DOWN * 2)
        b1 = Square(side_length=1.0, color=BLUE).next_to(wall, RIGHT, buff=1)
        b2 = Square(side_length=2.0, color=RED).next_to(b1, RIGHT, buff=2)
        self.play(Create(wall), Create(floor))
        self.play(FadeIn(b1), FadeIn(b2))
        counter = Integer(0).to_corner(UP + RIGHT)
        self.play(Create(counter))
        self.wait(1)