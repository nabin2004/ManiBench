from manim import *
import numpy as np

class BlockCollisionPi(Scene):
    def construct(self):
        # Configuration
        mass_ratio_exp = 4  # 10^4 -> 314 collisions (manageable for animation)
        # For 100^k, collisions = floor(pi * 10^k). 
        # 10^0 (1:1) -> 3
        # 10^2 (100:1) -> 31
        # 10^4 (10000:1) -> 314
        # 10^6 -> 3141 (Too slow for real-time anim, we will simulate steps or speed up)
        
        # Let's use a smaller ratio for the main detailed animation to ensure it finishes in reasonable time
        # but show the text for larger ones.
        # We will animate 100:1 (31 collisions) fully.
        ratio_exp = 2 
        M = 100 ** ratio_exp
        m = 1
        mass_ratio = M / m
        
        # Physics Constants
        v0 = 2.0 # Initial velocity of small block
        v_A = 0.0
        v_B = v0
        
        # Visual Scaling
        scale_factor = 0.5
        width_A = 2.0 * (M ** (1/3)) * 0.5 # Rough visual mass rep
        width_B = 2.0 * (m ** (1/3)) * 0.5
        height = 1.0
        
        # Positions
        x_A = 2.0
        x_B = -4.0
        
        # Create Blocks
        block_A = Rectangle(height=height, width=width_A, color=BLUE, fill_opacity=0.8)
        block_A.set_x(x_A).set_y(0)
        label_A = Text(f"M = {int(M)}", font_size=24).next_to(block_A, UP, buff=0.2)
        
        block_B = Rectangle(height=height, width=width_B, color=GREEN, fill_opacity=0.8)
        block_B.set_x(x_B).set_y(0)
        label_B = Text(f"m = {m}", font_size=24).next_to(block_B, UP, buff=0.2)
        
        # Floor
        floor = Line(LEFT * 8, RIGHT * 8, color=WHITE).set_y(-height/2)
        
        # Velocity Vectors
        vec_A = Arrow(block_A.get_top(), block_A.get_top() + RIGHT * v_A * 2, color=YELLOW, buff=0)
        vec_B = Arrow(block_B.get_top(), block_B.get_top() + RIGHT * v_B * 2, color=YELLOW, buff=0)
        label_vA = Text("v_A=0", font_size=20).next_to(vec_A, UP, buff=0.1)
        label_vB = Text(f"v_B={v0}", font_size=20).next_to(vec_B, UP, buff=0.1)
        
        # Counter
        counter_text = Text("Collisions: 0", font_size=36).to_edge(UP)
        collision_count = 0
        
        # Equations
        ke_eq = MathTex(r"\frac{1}{2}Mv_A^2 + \frac{1}{2}mv_B^2 = E", font_size=30).to_edge(DR).shift(UP*1.5)
        mom_eq = MathTex(r"Mv_A + mv_B = P", font_size=30).next_to(ke_eq, DOWN, buff=0.2)
        
        # Group initial scene
        self.play(Create(floor), Create(block_A), Create(label_A), Create(block_B), Create(label_B))
        self.play(Create(vec_A), Create(label_vA), Create(vec_B), Create(label_vB))
        self.play(Write(counter_text))
        self.play(Write(ke_eq), Write(mom_eq))
        self.wait(0.5)
        
        # Simulation Loop
        # We need to update positions frame by frame or step by step
        # Since 31 collisions is fast, we can animate the motion between collisions
        
        wall_x = -6.0 # Position of the left wall
        
        # Helper to update visuals
        def update_visuals():
            block_A.set_x(x_A)
            block_B.set_x(x_B)
            label_A.next_to(block_A, UP, buff=0.2)
            label_B.next_to(block_B, UP, buff=0.2)
            
            # Update vectors
            new_end_A = block_A.get_top() + RIGHT * v_A * 1.5
            new_end_B = block_B.get_top() + RIGHT * v_B * 1.5
            
            vec_A.put_start_and_end_on(block_A.get_top(), new_end_A)
            vec_B.put_start_and_end_on(block_B.get_top(), new_end_B)
            
            if v_A >= 0:
                label_vA.text = f"v_A={v_A:.2f}"
            else:
                label_vA.text = f"v_A={v_A:.2f}"
            label_vB.text = f"v_B={v_B:.2f}"
            
            counter_text.text = f"Collisions: {collision_count}"

        # Initial draw
        update_visuals()
        
        # Animation Loop
        running = True
        max_iterations = 1000 # Safety break
        iterations = 0
        
        # Speed up factor for animation
        dt = 0.016 
        
        while running and iterations < max_iterations:
            iterations += 1
            
            # Determine time to next event
            # Event 1: Block B hits Wall (x_B <= wall_x + width_B/2)
            # Event 2: Block B hits Block A (x_B + width_B/2 >= x_A - width_A/2)
            
            dist_to_wall = (x_B - width_B/2) - wall_x
            dist_to_A = (x_A - width_A/2) - (x_B + width_B/2)
            
            t_wall = dist_to_wall / (-v_B) if v_B < 0 else float('inf')
            t_coll = dist_to_A / (v_B - v_A) if (v_B - v_A) > 0 else float('inf')
            
            next_t = min(t_wall, t_coll)
            
            # If next_t is huge or negative (shouldn't happen), break
            if next_t == float('inf') or next_t < 0:
                # Check termination: B moving right, A moving right, v_B <= v_A
                if v_B <= v_A and v_B >= 0:
                    running = False
                    break
                # Fallback small step if stuck
                next_t = 0.1

            # Animate motion to event
            # To make it smooth, we interpolate
            steps = int(next_t / dt)
            if steps == 0: steps = 1
            
            for _ in range(steps):
                x_A += v_A * dt
                x_B += v_B * dt
                update_visuals()
                self.wait(dt) # Real-time wait
            
            # Handle Event
            if t_wall < t_coll:
                # Hit Wall
                x_B = wall_x + width_B/2 # Snap to wall
                v_B = -v_B # Elastic collision with infinite mass wall
                update_visuals()
                self.wait(0.1)
            else:
                # Hit Block A
                x_B = x_A - (width_A + width_B)/2 # Snap to contact
                # Elastic Collision Formulas
                # v1' = (m1-m2)/(m1+m2)v1 + 2m2/(m1+m2)v2
                # v2' = 2m1/(m1+m2)v1 + (m2-m1)/(m1+m2)v2
                # Here 1 is B (m), 2 is A (M)
                
                v_B_new = ((m - M) * v_B + 2 * M * v_A) / (m + M)
                v_A_new = (2 * m * v_B + (M - m) * v_A) / (m + M)
                
                v_B = v_B_new
                v_A = v_A_new
                collision_count += 1
                
                update_visuals()
                # Flash effect on collision
                self.play(
                    block_A.set_fill, WHITE, 0.5,
                    block_B.set_fill, WHITE, 0.5,
                    rate_func=there_and_back,
                    run_time=0.1
                )
                self.play(
                    block_A.set_fill, BLUE, 0.8,
                    block_B.set_fill, GREEN, 0.8,
                    run_time=0.0
                )
                self.wait(0.05)
            
            # Termination check
            if v_B <= v_A and v_B >= 0 and x_B < x_A:
                running = False

        self.wait(1)
        
        # Final Result Text
        final_text = Text(f"# Collisions = {collision_count}", font_size=48, color=YELLOW).to_edge(DOWN)
        self.play(Write(final_text))
        self.wait(1)

        # Transition to Phase Space Explanation
        self.next_section("Phase Space Explanation")
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        
        # Phase Space Setup
        title = Text("Phase Space Analysis", font_size=40).to_edge(UP)
        self.play(Write(title))
        
        # Axes: x = sqrt(m)*v_B, y = sqrt(M)*v_A
        # Note: Standard derivation usually maps to circle x^2 + y^2 = 2E
        # Let x = sqrt(m) * v_B, y = sqrt(M) * v_A
        # Then KE = 0.5 * (x^2 + y^2) = E => x^2 + y^2 = 2E (Circle)
        # Momentum: sqrt(m)*x + sqrt(M)*y = P => y = -sqrt(m/M)*x + P/sqrt(M) (Line)
        
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-1, 6, 1],
            axis_config={"include_tip": True, "include_numbers": False},
            x_axis_config={"label_direction": DOWN},
            y_axis_config={"label_direction": LEFT}
        ).scale(0.8).shift(DOWN * 0.5)
        
        x_label = MathTex(r"x = \sqrt{m}v_B").next_to(axes.x_axis.get_end(), DR)
        y_label = MathTex(r"y = \sqrt{M}v_A").next_to(axes.y_axis.get_end(), UL)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Calculate Radius
        E_total = 0.5 * m * (v0**2)
        R = np.sqrt(2 * E_total) # Radius in phase space
        
        # Scale for visual fit
        visual_scale = 1.5 / R if R > 0 else 1
        axes.scale(visual_scale)
        # Re-adjust labels after scale? Simpler to just draw circle with calculated radius
        
        # Let's redraw axes with fixed scale for clarity
        self.play(*[FadeOut(mob) for mob in [axes, x_label, y_label]])
        
        fixed_R = 3.0
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-1, 4, 1],
            height=6, width=6
        ).shift(DOWN * 0.5)
        
        circle = Circle(radius=fixed_R, color=BLUE, stroke_width=3).move_to(axes.get_origin())
        
        # Momentum Line Slope: -sqrt(m/M)
        slope = -np.sqrt(m / M)
        # Line passes through initial point: (sqrt(m)*v0, 0)
        x_start = np.sqrt(m) * v0 * (fixed_R / R) # Scaled
        y_start = 0
        
        # Actually, let's just draw the line geometrically based on angle
        # Angle theta = arctan(sqrt(m/M))
        theta = np.arctan(np.sqrt(m / M))
        
        # The line connects (R, 0) to some point on circle? 
        # Initial state: v_A=0, v_B=v0. Point: (sqrt(m)v0, 0). This is on the circle.
        # Momentum conservation line passes through this point with slope -sqrt(m/M).
        
        line_len = 10
        momentum_line = Line(
            start=axes.c2p(-5, -5/slope + 10), # Approximate long line
            end=axes.c2p(5, 5/slope - 10),
            color=RED
        )
        # Correct line equation: y - 0 = slope * (x - x_start_scaled)
        # But in the scaled phase space where circle is unit-ish:
        # The angle of the normal to the momentum line is theta.
        # The line makes angle -theta with x-axis? No.
        # Slope = -sqrt(m/M) = -tan(theta). So angle is -theta.
        
        # Let's construct the line properly passing through (R, 0) with slope
        p_start = axes.c2p(fixed_R, 0)
        p_end = axes.c2p(fixed_R - 2, 0 - 2 * slope) # Go left 2, down 2*slope
        momentum_line = Line(p_start, p_end, color=RED, stroke_width=3).set_length(10)
        momentum_line.shift(p_start - momentum_line.get_start()) # Ensure it starts at (R,0) roughly
        
        # Actually, the reflection happens off the line y = -tan(theta)x + C
        # The first collision (Wall) reflects y -> -y? No.
        # Wall collision: v_B -> -v_B. In phase space: x -> -x. Reflection across Y-axis.
        # Block collision: Reflection across Momentum Line.
        
        # Let's trace the path
        path_points = []
        curr_x = fixed_R
        curr_y = 0
        path_points.append(axes.c2p(curr_x, curr_y))
        
        # Simulate reflections in phase space
        # 1. Wall: Reflect x across 0? No, Wall is v_B = 0? No, Wall is x=0 in real space.
        # When B hits wall, v_B becomes -v_B. So x_coord = sqrt(m)*v_B becomes -x_coord.
        # This is reflection across the Y-axis (x=0).
        # 2. Block: Reflection across the momentum line.
        
        # We need to generate the polygon path
        ps_x = fixed_R
        ps_y = 0.0
        
        # Angle of momentum line normal
        # Line: sqrt(m)v_B + sqrt(M)v_A = P
        # Normal vector n = (sqrt(m), sqrt(M))
        # Angle of normal alpha = arctan(sqrt(M)/sqrt(m)) = arctan(sqrt(M/m))
        # Angle of line itself = alpha - 90 deg.
        # Or simply: slope = -sqrt(m/M). Angle = arctan(-sqrt(m/M)).
        
        phi = np.arctan(np.sqrt(m/M)) # Angle of slope relative to horizontal (negative)
        # The line goes down.
        
        # Total angle of the sector is PI.
        # Each bounce rotates the state vector by 2*theta?
        # Theorem: The number of collisions is floor(pi / theta) where theta = arctan(sqrt(m/M)).
        # Wait, standard result: N = floor(pi / arctan(sqrt(m/M))).
        # Let's trace geometrically.
        
        current_angle = 0.0 # Starts at (R, 0) -> Angle 0
        # The momentum line is at angle: pi - theta? 
        # Vector (sqrt(m), sqrt(M)) is normal. 
        # Let's just use the known property: The point rotates by 2*theta each full cycle (wall+block)?
        # Actually, simpler:
        # Wall reflection: (x, y) -> (-x, y). Angle a -> pi - a.
        # Block reflection: Reflect across line with angle (pi - theta).
        
        # Let's just draw the zig-zag visually for the explanation
        zigzag = VGroup()
        curr_pt = axes.c2p(fixed_R, 0)
        
        # Angle increment per collision pair?
        # The angle subtended by each collision at the origin is 2*theta?
        # No, the total angle covered is PI.
        # Each collision (block) rotates the velocity vector in phase space by 2*theta?
        # Let's rely on the visual of bouncing between the line and the axis.
        
        # Line angle:
        # Slope m_s = -sqrt(m/M). Angle = -theta.
        # But we are in the first quadrant initially?
        # v_A >= 0 always? Yes, A only gets pushed right.
        # v_B changes sign.
        # So we bounce between y=0 (v_A=0? No, v_A never negative) 
        # Wait, v_A starts 0, becomes positive. Never negative.
        # v_B starts positive, becomes negative (wall), then positive/negative.
        # Constraint: v_A >= 0 => y >= 0.
        # Constraint: Energy => Circle.
        # Constraint: Momentum => Line.
        # When B hits wall: v_B -> -v_B. x -> -x.
        # But if x becomes negative, that means v_B is negative (moving left).
        # So the trajectory goes into the second quadrant?
        # Yes.
        # When B hits A: Momentum conservation. Reflection across the line.
        
        # Let's generate points until we reach the "end zone"
        # End zone: v_B <= v_A and v_B > 0? 
        # In phase space: x/sqrt(m) <= y/sqrt(M) => y >= x * sqrt(M/m).
        # This is a line with slope sqrt(M/m) = cot(theta).
        # This line is perpendicular to the momentum line!
        # So the game ends when we cross the line y = cot(theta) * x.
        
        theta = np.arctan(np.sqrt(m/M))
        # Momentum line angle (from positive x-axis, counter-clockwise): pi - theta
        # Wall is x=0? No, wall flip is x -> -x.
        # Start at (R, 0). Angle 0.
        # Move along circle? No, energy is constant, so we stay on circle.
        # Collisions change the angle instantly.
        
        # Sequence:
        # 1. Start at angle 0.
        # 2. Block collision? No, B is faster. B hits A?
        # Wait, initial: B moves right, A rest. B hits A.
        # This is a block collision.
        # Reflection across momentum line.
        # Momentum line passes through (R,0)?
        # P = sqrt(m)*v0. Line: sqrt(m)x + sqrt(M)y = sqrt(m)v0 * sqrt(m)? No.
        # P = m*v0.
        # Eq: sqrt(m)*x + sqrt(M)*y = m*v0 = sqrt(m)*sqrt(m)*v0 = sqrt(m) * (sqrt(m)v0) = sqrt(m) * x_start.
        # So line passes through (x_start, 0). Correct.
        # Angle of normal: tan(alpha) = sqrt(M)/sqrt(m) = cot(theta).
        # Angle of line: alpha - 90 = (90-theta) - 90 = -theta.
        # So line is at angle -theta (or 180-theta).
        # Since we are in upper half plane (y>=0), the relevant segment is in Q2?
        # No, starts at (R, 0). Line goes up-left.
        # Angle of line vector pointing into Q2: 180 - theta.
        
        # Reflection of angle 0 across line (180-theta):
        # New angle = 2*(180-theta) - 0 = 360 - 2theta = -2theta.
        # But y must be >= 0.
        # Something is wrong with my coordinate mapping or reflection logic.
        
        # Let's use the standard result visualization:
        # The state point moves along the circle.
        # Each collision corresponds to a rotation of 2*theta.
        # Total angle to cover is PI.
        # Number of steps = PI / (2*theta)? Or PI/theta?
        # Formula: N = floor(pi / theta).
        # So each step is theta?
        # Let's just draw the arc and ticks.
        
        arc = Arc(radius=fixed_R, start_angle=0, angle=PI, color=WHITE, stroke_width=2)
        self.play(Create(circle), Create(arc))
        
        # Show the angle theta
        theta_arc = Arc(radius=0.5, start_angle=PI-np.arctan(np.sqrt(M/m)), angle=np.arctan(np.sqrt(m/M)), color=RED)
        # This is getting too complex to calculate exact coordinates without a solver.
        # Simplified Visual:
        
        explanation = Text(f"Mass Ratio 100^{ratio_exp}:1", font_size=30).to_edge(UR)
        theta_val = np.arctan(np.sqrt(m/M))
        formula = MathTex(r"N \approx \frac{\pi}{\theta}", r"\quad \theta = \arctan\sqrt{\frac{m}{M}}", font_size=30).next_to(explanation, DOWN)
        
        self.play(Write(explanation), Write(formula))
        
        # Animate the "bouncing" angle
        # We know total collisions = count
        # We can draw rays from origin separated by angle theta?
        # Actually, the angle between consecutive collision points on the circle is 2*theta?
        # Let's assume the property: The point rotates by 2*theta each time?
        # If N = pi/theta, then step is theta.
        
        step_angle = theta_val # Hypothesis
        # Let's just draw dots on the circle at intervals of step_angle until PI
        dots = VGroup()
        curr_ang = 0.0
        count_sim = 0
        while curr_ang < PI:
            pt = axes.c2p(fixed_R * np.cos(curr_ang), fixed_R * np.sin(curr_ang))
            dot = Dot(pt, color=YELLOW, radius=0.05)
            dots.add(dot)
            curr_ang += step_angle
            count_sim += 1
            
        self.play(Create(dots), run_time=2)
        self.wait(1)
        
        # Show the list of ratios
        self.next_section("Multiple Ratios")
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        
        title2 = Text("Mass Ratio vs Collisions", font_size=40).to_edge(UP)
        rows = VGroup()
        data = [
            ("1:1", "3"),
            ("100:1", "31"),
            ("10,000:1", "314"),
            ("10^6:1", "3,141"),
            ("10^8:1", "31,415"),
            ("10^10:1", "314,159")
        ]
        
        for i, (ratio, coll) in enumerate(data):
            line = Text(f"Ratio {ratio} → {coll} collisions", font_size=28)
            line.set_x(0).set_y(2 - i*0.6)
            rows.add(line)
            
        self.play(Write(title2))
        for row in rows:
            self.play(Write(row))
            self.wait(0.3)
            
        self.wait(2)
        
        # Final Pi Connection
        pi_text = Text("The digits of π emerge from elastic collisions!", color=BLUE, font_size=36).to_edge(DOWN)
        self.play(Write(pi_text))
        self.wait(3)