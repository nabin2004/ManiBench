from manim import *

class ManiBenchEvaluationScene(Scene):
    def construct(self):
        title = Title('ManiBench Evaluation Benchmark')
        self.play(Write(title))
        line = Line(LEFT * 4, RIGHT * 4, color=BLUE)
        dot = Dot(color=YELLOW).move_to(line.get_start())
        self.play(Create(line), FadeIn(dot))
        self.play(dot.animate.move_to(line.get_end()))
        formula = MathTex(r'E = mc^2').next_to(line, DOWN)
        self.play(Write(formula))
        self.wait(1)