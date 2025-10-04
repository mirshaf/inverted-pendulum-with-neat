import pymunk
import math

WIDTH, HEIGHT = 1000, 700

class Pendulum:
    def __init__(self, space):
        # Create anchor body as KINEMATIC so we can control its position
        self.pivot_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.pivot_body.position = (WIDTH/2, HEIGHT/2 - 50)

        self.bob_body = pymunk.Body() # The weighted object at the end of the pendulum
        self.bob_body.position = (WIDTH/2, HEIGHT/2 + 50)
        circle_shape = pymunk.Circle(self.bob_body, 20, (0, 0))
        circle_shape.friction = 1
        circle_shape.mass = 20
        circle_shape.elasticity = 0.95
        suspension = pymunk.PinJoint(self.bob_body, self.pivot_body, (0, 0), (0, 0)) # The cord that holds the bob and suspends it from a fixed point

        shape_filter = pymunk.ShapeFilter(group=1) # Make sure pendulums don't collide with each other
        circle_shape.filter = shape_filter
        
        # Calculate pendulum length
        self.pendulum_length = math.sqrt(
            (self.bob_body.position.x - self.pivot_body.position.x)**2 + 
            (self.bob_body.position.y - self.pivot_body.position.y)**2
        )

        self.everything_in_space = [circle_shape, self.bob_body, suspension]
        space.add(*self.everything_in_space)

    def get_angular_velocity(self):
        # Calculate tangential velocity (component perpendicular to the pendulum arm)
        dx = self.bob_body.position.x - self.pivot_body.position.x
        dy = self.bob_body.position.y - self.pivot_body.position.y
        pendulum_length = math.sqrt(dx**2 + dy**2)
        if pendulum_length == 0:
            return 0

        # Calculate the velocity of the circle relative to the pivot
        rel_velocity = (
            self.bob_body.velocity.x - self.pivot_body.velocity.x,
            self.bob_body.velocity.y - self.pivot_body.velocity.y
        )
        
        # Unit vector along the pendulum arm
        arm_direction = (dx/pendulum_length, dy/pendulum_length)
        
        # Unit vector perpendicular to the pendulum arm (tangential direction)
        tangential_direction = (-arm_direction[1], arm_direction[0])
        
        # Dot product of velocity with tangential direction gives tangential speed
        tangential_speed = (
            rel_velocity[0] * tangential_direction[0] + 
            rel_velocity[1] * tangential_direction[1]
        )
        
        # Angular velocity = tangential speed / radius
        return tangential_speed / pendulum_length
    
    def get_sensory_data(self):
        # Calculate the angle of the pendulum relative to positive x-axis
        dx = self.bob_body.position.x - self.pivot_body.position.x
        dy = self.bob_body.position.y - self.pivot_body.position.y
        angle = math.atan2(dy, dx)  # Returns angle in radians
        
        # Normalize inputs for the neural net
        normalized_pivot_x = (self.pivot_body.position.x - WIDTH/6) / (WIDTH - WIDTH/3) * 2 - 1  # Map to [-1, 1]
        normalized_angle = angle / math.pi  # Map to [-1, 1]
        normalized_angular_velocity = self.get_angular_velocity() / math.pi  # Scale down
        
        return normalized_pivot_x, normalized_angle, normalized_angular_velocity
  