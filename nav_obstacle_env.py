__docformat__ = "reStructeredText"

import random
import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
from numpy import pi
from skimage.transform import rescale

random.seed(2)

def limit_velocity(body, gravity, damping, dt):
        max_velocity = 100
        pymunk.Body.update_velocity(body, gravity, body.damping, dt)
        l = body.velocity.length
        if l > max_velocity:
            scale = max_velocity / l
            body.velocity = body.velocity * scale

def custom_damping(body, gravity, damping, dt):
    pymunk.Body.update_velocity(body, gravity, body.damping, dt)

def distance(pos1, pos2):
    return abs(pos1 - pos2)

class Nav_Obstacle_Env(object):
    def __init__(self, state_type='vision') -> None:
        self.state_type = state_type

        # Space
        self._space = pymunk.Space()
        self._space.gravity = (0.0, 0.0)

        # Physics
        # Time step
        self._dt = 1.0 / 240.0
        # Number of physics steps per screen frame
        self._physics_steps_per_frame = 1

        # pygame
        pygame.display.init()
        pygame.font.init()
        self.screen_size = (600,600)
        self._screen = pygame.display.set_mode(self.screen_size)
        self._clock = pygame.time.Clock()

        self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)
        self.pxarr = pygame.PixelArray(self._draw_options.surface)

        # Static barrier walls (lines) that the balls bounce off of
        self._add_static_scenery()

        # The agent to be controlled
        agent_y_pos = random.randint(50,self.screen_size[1]-50)
        self._agent = self._create_agent(vertices=((-25,-25), (-25,25), (25,25), (25,-25)), mass=10, position=(self.screen_size[0]*0.75, agent_y_pos), damping=0.99)

        if self.state_type == 'vision':
            self.state = np.zeros((1, self.screen_size[0], self.screen_size[1])).astype(int)
            self.get_vision_state()
        else:
            self.left_sensor_data = self._space.point_query_nearest(point=self._agent['robot'].local_to_world((-25, -25)), max_distance=30, shape_filter=pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b101))
            self.right_sensor_data = self._space.point_query_nearest(point=self._agent['robot'].local_to_world((25, -25)), max_distance=30, shape_filter=pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b101))
            self.state = np.zeros((1,2))
            self.get_state()

        # Agent cumulative rewards
        self.reward = 0
        self.reward_from_last_action = 0

        # Rewards
        self.collision_penalty = 0.25
        self.goal_reward = 1
        self.partial_rewards_scale = 0.5

        # Available actions
        self.available_actions = ['forward', 'backward', 'turn_cw', 'turn_ccw']

        self.left_sensor_data = None
        self.right_sensor_data = None

        self.goal_position = (25,75)
        self.initial_agent_dist = distance(self._agent['robot'].position, self.goal_position)
        self.initial_agent_pos = self._agent['robot'].position

        # Collision Handling
        # Robot: 0, Obstacles: 1, Goal: 2
        self.collision_occuring = False
        self.handler = self._space.add_collision_handler(0,1)
        self.handler.begin = self.collision_begin
        self.handler.separate = self.collision_end
        self.goal_handler = self._space.add_collision_handler(0,2)
        self.goal_handler.begin = self.collision_goal

        # Execution control and time unitl the next ball spawns
        self._running = True
        self._done = False

    def _add_static_scenery(self) -> None:
        """
        Create the static bodies
        :return: None
        """
        static_body = self._space.static_body

        num_obstacles = random.randint(1,4)
        static_obstacles = []
        obstacles_info = []
        for i in range(num_obstacles):
            obs_pos_x = random.randint(130,375)
            obs_pos_y = random.randint(130,470)
            obs_size_x = random.randint(25,50)
            obs_size_y = random.randint(25,50)
            if obstacles_info is not None:
                pass #TODO
            obstacles_info.append([obs_pos_x, obs_pos_y, obs_size_x, obs_size_y])
            static_obstacles.append(pymunk.Poly(static_body, ((obs_pos_x-obs_size_x,obs_pos_y-obs_size_y), (obs_pos_x+obs_size_x,obs_pos_y-obs_size_y), (obs_pos_x+obs_size_x, obs_pos_y+obs_size_y), (obs_pos_x-obs_size_x,obs_pos_y+obs_size_y))),)
            
        for obstacle in static_obstacles:
            obstacle.elasticity = 0
            obstacle.friction = 1
            obstacle.collision_type = 1
            obstacle.filter = pymunk.ShapeFilter(categories=0b10)
        self._space.add(*static_obstacles)

        static_border = [
            pymunk.Segment(static_body, (0,0), (0,600), 1),
            pymunk.Segment(static_body, (0,0), (600,0), 1),
            pymunk.Segment(static_body, (599,0), (599,599), 1),
            pymunk.Segment(static_body, (0,599), (599,599), 1),
        ]
        for line in static_border:
            line.elasticity = 0.95
            line.friction = 0.9
            line.collision_type = 1
            line.filter = pymunk.ShapeFilter(categories=0b10)
        self._space.add(*static_border)

        static_goal = [
            pymunk.Poly(static_body, ((0,0), (50,0), (50,150), (0, 150))),
        ]
        


        static_goal[0].color = (0, 255, 0, 255)
        static_goal[0].collision_type = 2
        static_goal[0].filter = pymunk.ShapeFilter(categories=0b101)

        self._space.add(*static_goal)

    def _create_agent(self, vertices: list[tuple[int, int]], mass: float, position: tuple[int] = (0,0), elasticity: float = 0, friction: float = 1, damping: float = 1.0) -> pymunk.Poly:
        """
        Create the agent
        :return: Pymunk Polygon
        """
        robot_body = pymunk.Body()
        robot_body.position = position
        robot_body.damping = damping

        robot = pymunk.Poly(robot_body, vertices)
        robot.label = 'robot'
        robot.mass = mass
        robot.elasticity = elasticity
        robot.friction = friction
        robot.collision_type = 0
        robot.filter = pymunk.ShapeFilter(categories=0b1)
        self._space.add(robot_body, robot)
        robot_body.angle = round(random.randrange(314)/100, 2)

        wheel_1_body = pymunk.Body()
        wheel_1_body.position = robot_body.local_to_world((-30, 15))
        wheel_1_body.angle = robot_body.angle
        wheel_2_body = pymunk.Body()
        wheel_2_body.position = robot_body.local_to_world((30, 15))
        wheel_2_body.angle = robot_body.angle

        wheel_1 = pymunk.Poly(wheel_1_body, ([-5, -10], [5, -10], [5, 10], [-5, 10]))
        wheel_2 = pymunk.Poly(wheel_2_body, ([-5, -10], [5, -10], [5, 10], [-5, 10]))
        for i, wheel in enumerate([wheel_1, wheel_2]):
            wheel.body.damping = damping
            wheel.body.velocity_func = limit_velocity
            wheel.body.latch = False
            wheel.body.forward = True

            wheel.label = f'wheel_{i+1}'
            wheel.mass = 2
            wheel.elasticity = 0
            wheel.friction = 100000
            wheel.collision_type = 0
            wheel.color = (0, 0, 0, 255)
            wheel.filter = pymunk.ShapeFilter(categories=0b1)
            self._space.add(wheel.body, wheel)

        joint_1 = pymunk.constraints.PinJoint(robot.body, wheel_1.body, (-25,25), (5,10))
        joint_2 = pymunk.constraints.PinJoint(robot.body, wheel_1.body, (-25,5), (5,-10))
        joint_5 = pymunk.constraints.PinJoint(wheel_1.body, wheel_2.body, (-5,0), (5,0))
        joint_3 = pymunk.constraints.PinJoint(robot.body, wheel_2.body, (25,25), (-5,10))
        joint_4 = pymunk.constraints.PinJoint(robot.body, wheel_2.body, (25,5), (-5,-10))
        joint_6 = pymunk.constraints.PinJoint(wheel_1.body, wheel_2.body, (-5,0), (5,0))
        
        self._space.add(joint_1, joint_2, joint_3, joint_4, joint_5, joint_6)
        return {'robot': robot_body, 'wheel_1': wheel_1_body, 'wheel_2': wheel_2_body}

    def run(self) -> None:
        """
        The mail loop of the game
        :return: None
        """
        # Main Loop
        while self._running:
            # Progress time forward
            for x in range(self._physics_steps_per_frame):
                self._space.step(self._dt)

            action = self._process_events()
            self._actions(action)
            self._update()
            self._clear_screen()
            self._draw_objects()
            if self.state_type == 'vision':
                self.get_vision_state()
            else:
                self.get_state()

            pygame.display.flip()
            # Delay fixed time between frames
            self._clock.tick(110)
            pygame.display.set_caption("fps: " + str(self._clock.get_fps()))

            if action is not None:
                self.reward +=  self.reward_from_last_action
                self.reward_from_last_action = 0
            
            # Calculate reward
            self.reward_from_last_action += self.get_reward(True if action is not None else False)
            
            if self._done:
                self.reset()
    
    def step(self, action):
        """
        Progress the simulation one time step
        inputs: action
        outputs: state, reward, done, info
        """
        # Progress time forward
        for x in range(self._physics_steps_per_frame):
            self._space.step(self._dt)

        self._process_events()
        self._actions(action)
        self._update()
        self._clear_screen()
        self._draw_objects()
        if self.state_type == 'vision':
            self.get_vision_state()
        else:
            self.get_state()
        
        pygame.display.flip()

        # Delay fixed time between frames
        self._clock.tick(230)
        pygame.display.set_caption("fps: " + str(self._clock.get_fps()))

        if action is not None:
            self.reward += self.reward_from_last_action
            self.reward_from_last_action = 0

        # Calculate reward
        self.reward_from_last_action += self.get_reward(True if action is not None else False)

        # Items to return
        state = self.state
        reward = self.reward_from_last_action
        done = self._done
        # robot_distance = distance(self.initial_agent_pos, self._agent['robot'].position)
        # self.initial_agent_pos = self._agent['robot'].position
        # ministeps = robot_distance / 0.25
        info = { # To match SAM
            # 'ministeps': ministeps,
            'inactivity': None,
            'cumulative_cubes': 0,
            'cumulative_distance': 0,
            'cumulative_reward': self.reward
        }
        return state, reward, done, info

    def _process_events(self) -> str:
        """
        Handle game and events like keyboard input. Call once per frame only.
        :return: action
        """
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._running = False

            elif event.type == pygame.KEYDOWN:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]:
                    # self._actions('forward')
                    action = 'forward'
                elif keys[pygame.K_DOWN]:
                    # self._actions('backward')
                    action = 'backward'
                elif keys[pygame.K_LEFT]:
                    # self._actions('turn_ccw')
                    action = 'turn_ccw'
                elif keys[pygame.K_RIGHT]:
                    # self._actions('turn_cw')
                    action = 'turn_cw'
        return action
    
    def _update(self):
        # print(pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS()), pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b10), end='\r')
        self.left_sensor_data = self._space.point_query_nearest(point=self._agent['robot'].local_to_world((-25, -25)), max_distance=30, shape_filter=pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b101))
        self.right_sensor_data = self._space.point_query_nearest(point=self._agent['robot'].local_to_world((25, -25)), max_distance=30, shape_filter=pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b101))

        for sensor in [self.left_sensor_data, self.right_sensor_data]:
            if sensor is not None:
                # self._agent['robot'].score -= round((30-sensor[2])/60, 2)
                # print(round((30-sensor[2])/60, 2), end='\r')
                pass

        if self._agent['wheel_1'].latch:
            if self._agent['wheel_1'].forward:
                self._agent['wheel_1'].velocity = -abs(self._agent['wheel_1'].velocity) * self._agent['wheel_1'].rotation_vector.perpendicular()
            else:
                self._agent['wheel_1'].velocity = abs(self._agent['wheel_1'].velocity) * self._agent['wheel_1'].rotation_vector.perpendicular()

        if self._agent['wheel_2'].latch:
            if self._agent['wheel_2'].forward:
                self._agent['wheel_2'].velocity = -abs(self._agent['wheel_2'].velocity) * self._agent['wheel_2'].rotation_vector.perpendicular()
            else:
                self._agent['wheel_2'].velocity = abs(self._agent['wheel_2'].velocity) * self._agent['wheel_2'].rotation_vector.perpendicular()

        for key in self._agent:
            if key != 'robot':
                if self._agent[key].position.x > 600:
                    self._agent[key].position = (590, self._agent[key].position.y)
                
                elif self._agent[key].position.x < 0:
                    self._agent[key].position = (10, self._agent[key].position.y)

                if self._agent[key].position.y > 600:
                    self._agent[key].position = (self._agent[key].position.x, 590)
                
                elif self._agent[key].position.y < 0:
                    self._agent[key].position = (self._agent[key].position.x, 10)

                if (self._agent[key].position - self._agent['robot'].position).length > 34:
                    if key == 'wheel_1':
                        self._agent[key].position = self._agent['robot'].local_to_world((-30, 15))
                    else:
                        self._agent[key].position = self._agent['robot'].local_to_world((30, 15))

    def _clear_screen(self) -> None:
        """
        Clears the screen.
        :return: None
        """
        self._screen.fill(pygame.Color("white"))

    def _draw_objects(self) -> None:
        """
        Draw the objects.
        :return: None
        """
        self._space.debug_draw(self._draw_options)
    
    def _actions(self, action) -> None:
        """
        action: 'forward', 'backward', 'turn_cw', 'turn_ccw'
        :return: None
        """
        if action == 'forward':
            self._agent['wheel_1'].velocity += self._agent['wheel_1'].rotation_vector.perpendicular() * -50
            self._agent['wheel_1'].latch = True
            self._agent['wheel_1'].forward = True
            self._agent['wheel_2'].velocity += self._agent['wheel_2'].rotation_vector.perpendicular() * -50
            self._agent['wheel_2'].latch = True
            self._agent['wheel_2'].forward = True

        elif action == 'backward':
            self._agent['wheel_1'].velocity += self._agent['wheel_1'].rotation_vector.perpendicular() * 50
            self._agent['wheel_1'].latch = True
            self._agent['wheel_1'].forward = False
            self._agent['wheel_2'].velocity += self._agent['wheel_2'].rotation_vector.perpendicular() * 50
            self._agent['wheel_2'].latch = True
            self._agent['wheel_2'].forward = False

        elif action == 'turn_cw':
            self._agent['wheel_1'].velocity += self._agent['wheel_1'].rotation_vector.perpendicular() * -50
            self._agent['wheel_1'].latch = True
            self._agent['wheel_1'].forward = True
            self._agent['wheel_2'].velocity += self._agent['wheel_2'].rotation_vector.perpendicular() * 50
            self._agent['wheel_2'].latch = True
            self._agent['wheel_2'].forward = False

        elif action == 'turn_ccw':
            self._agent['wheel_1'].velocity += self._agent['wheel_1'].rotation_vector.perpendicular() * 50
            self._agent['wheel_1'].latch = True
            self._agent['wheel_1'].forward = False
            self._agent['wheel_2'].velocity += self._agent['wheel_2'].rotation_vector.perpendicular() * -50
            self._agent['wheel_2'].latch = True
            self._agent['wheel_2'].forward = True

    def collision_begin(self, arbiter, space, dummy):
        shapes = arbiter.shapes
        self.collision_occuring = True
        self._agent['wheel_1'].latch = False
        self._agent['wheel_2'].latch = False
        return True
    
    def collision_end(self, arbiter, space, dummy):
        self.collision_occuring = False
        return True
    
    def collision_goal(self, arbiter, space, dummy):
        shapes = arbiter.shapes
        self._done = True
        return True
    
    def get_state(self):
        """left/right sensor data and direction (angle in rad)"""
        state = []
        for sensor in [self.left_sensor_data, self.right_sensor_data]:
            if sensor is not None:
                state.append(sensor[2])
            else:
                state.append(100)
        state.append(self._agent['robot'].angle%(2*pi))
        self.state = np.array(state)
    
    def get_vision_state(self):
        """
        Gets integer pixel values from screen
        """
        x,y = self._agent['robot'].position
        x_low = round(x-100) if x-100 > 0 else 0
        x_high = round(x+100) if x+100 < 600 else 600
        y_low = round(y-100) if y-100 > 0 else 0
        y_high = round(y+100) if y+100 < 600 else 600

        x_idx_high = x_high-x_low if x_high == 600 else 200
        x_idx_low = 200-x_high if x_low == 0 else 0
        y_idx_high = y_high-y_low if y_high == 600 else 200
        y_idx_low = 200-y_high if y_low == 0 else 0
        
        # self.state = np.zeros((1,200,200))
        # self.state[0,x_idx_low:x_idx_high, y_idx_low:y_idx_high] = np.array(self.pxarr[x_low:x_high,y_low:y_high]).astype('uint8')
        self.state = np.array(self.pxarr).astype('uint8').transpose()
        self.state = np.resize(rescale(self.state, 0.5)*255, (1,int(self.screen_size[0]-2),int(self.screen_size[1]/2)))

    def get_reward(self, action_taken: bool):
        """
        Penalty for collision with walls
        Reward for pushing object into goal
        Partial reward/penalty for pushing object closer to / further from goal
        """
        reward = 0
        if action_taken:
            # reward -= self.action_penalty
            pass

        if self.collision_occuring:
            reward -= self.collision_penalty
        
        if self._done:
            reward += self.goal_reward

        dist = distance(self._agent['robot'].position, self.goal_position)
        dist_moved = self.initial_agent_dist - dist
        self.initial_agent_dist = dist
        reward += self.partial_rewards_scale*dist_moved

        return reward
    
    def reset(self):
        cumulative_reward = self.reward
        reward_from_last_action = self.reward_from_last_action
        self.__init__(state_type=self.state_type)
        self.reward = cumulative_reward
        self.reward_from_last_action = reward_from_last_action
        return self.state
    
    def close(self):
        self._running = False

        
def main():
    game = Nav_Obstacle_Env()
    game.run()

if __name__ == "__main__":
    main()
