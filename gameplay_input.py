import numpy as np
from dataclasses import dataclass, field
from raylib import *
from pyray import Vector3, Color, Vector2
import quat

# debug draw colors
COLOR_TRAJECTORY = Color(138, 18, 252)  # Purple
COLOR_VELOCITY = Color(247, 69, 220)     # Pink
COLOR_FACING = Color(255, 165, 0)        # Orange

@dataclass
class GameplayInput:
    # === Gamepad Input ===
    gamepad_stick_left: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    gamepad_stick_right: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    gamepad_trigger_left: float = 0.0
    gamepad_trigger_right: float = 0.0
    
    # === Control Mode ===
    control_type: str = 'uncontrolled'
    desired_strafe: bool = False
    
    # === Camera State ===
    camera_azimuth: float = 0.0
    camera_altitude: float = 0.0
    camera_distance: float = 4.0
    
    # === Character Simulation State ===
    current_position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    current_rotation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    current_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    current_angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    
    # === Movement Parameters ===
    movement_speed: float = 3.0
    
    # === Physics Parameters ===
    velocity_halflife: float = 0.2
    rotation_halflife: float = 0.2
    dt: float = 1.0 / 60.0
    
    # === Action Flags ===
    reset_requested: bool = False
    mode_switch_requested: str | None = None 
    
    # === Trajectory State ===
    trajectory_positions: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float32))
    trajectory_directions: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float32))
    root_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    
    # === Velocity Facing State ===
    facing_direction: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    
    @property
    def left_stick_magnitude(self) -> float:
        return float(np.linalg.norm(self.gamepad_stick_left))
    
    @property
    def right_stick_magnitude(self) -> float:
        return float(np.linalg.norm(self.gamepad_stick_right))
    
    @property
    def movement_direction_world(self) -> np.ndarray:
        stick_3d = np.array([self.gamepad_stick_left[0], 0.0, self.gamepad_stick_left[1]], dtype=np.float32)
        camera_rot = quat.from_angle_axis(self.camera_azimuth, np.array([0, 1, 0]))
        return quat.mul_vec(camera_rot, stick_3d)
    
    @property
    def facing_direction_world(self) -> np.ndarray | None:
        if self.right_stick_magnitude < 1e-3:
            return None
        stick_3d = np.array([self.gamepad_stick_right[0], 0.0, self.gamepad_stick_right[1]], dtype=np.float32)
        camera_rot = quat.from_angle_axis(self.camera_azimuth, np.array([0, 1, 0]))
        direction = quat.mul_vec(camera_rot, stick_3d)
        return direction / np.linalg.norm(direction)
    
    @property
    def effective_movement_speed(self) -> float:
        if self.gamepad_trigger_right > 0.5:
            return self.movement_speed * 1.5
        return self.movement_speed
    
    def update_from_gamepad(self, gamepad_id: int = 0, deadzone: float = 0.1):
        if not IsGamepadAvailable(gamepad_id):
            self.gamepad_stick_left = np.zeros(2, dtype=np.float32)
            self.gamepad_stick_right = np.zeros(2, dtype=np.float32)
            self.gamepad_trigger_left = 0.0
            self.gamepad_trigger_right = 0.0
            return
        
        # Left stick (movement)
        left_x = GetGamepadAxisMovement(gamepad_id, GAMEPAD_AXIS_LEFT_X)
        left_y = GetGamepadAxisMovement(gamepad_id, GAMEPAD_AXIS_LEFT_Y)
        left_mag = np.sqrt(left_x**2 + left_y**2)
        
        if left_mag > deadzone:
            scaled_mag = min(left_mag * left_mag, 1.0)
            self.gamepad_stick_left[0] = (left_x / left_mag) * scaled_mag
            self.gamepad_stick_left[1] = (left_y / left_mag) * scaled_mag
        else:
            self.gamepad_stick_left = np.zeros(2, dtype=np.float32)
        
        # Right stick (camera/look)
        right_x = GetGamepadAxisMovement(gamepad_id, GAMEPAD_AXIS_RIGHT_X)
        right_y = GetGamepadAxisMovement(gamepad_id, GAMEPAD_AXIS_RIGHT_Y)
        right_mag = np.sqrt(right_x**2 + right_y**2)
        
        if right_mag > deadzone:
            normalized_mag = min(right_mag * right_mag, 1.0)
            scaled_mag = normalized_mag * normalized_mag
            self.gamepad_stick_right[0] = (right_x / right_mag) * scaled_mag
            self.gamepad_stick_right[1] = (right_y / right_mag) * scaled_mag
        else:
            self.gamepad_stick_right = np.zeros(2, dtype=np.float32)
        
        # Triggers
        self.gamepad_trigger_left = GetGamepadAxisMovement(gamepad_id, GAMEPAD_AXIS_LEFT_TRIGGER)
        self.gamepad_trigger_right = GetGamepadAxisMovement(gamepad_id, GAMEPAD_AXIS_RIGHT_TRIGGER)
        
        # Buttons
        self.desired_strafe = IsGamepadButtonDown(gamepad_id, GAMEPAD_BUTTON_LEFT_TRIGGER_2)
    
    def update_from_keyboard(self):
        # Control mode switching
        self.mode_switch_requested = None
        if IsKeyPressed(KEY_ONE):
            self.mode_switch_requested = 'uncontrolled'
        elif IsKeyPressed(KEY_TWO):
            self.mode_switch_requested = 'trajectory'
        elif IsKeyPressed(KEY_THREE):
            self.mode_switch_requested = 'velocity_facing'
        
        # Actions
        self.reset_requested = IsKeyPressed(KEY_R)
    
    def apply_mode_switch(self):
        if self.mode_switch_requested is not None:
            self.control_type = self.mode_switch_requested
            self.mode_switch_requested = None
            return True
        return False
    
    def update_simulation_state(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        velocity: np.ndarray,
        angular_velocity: np.ndarray
    ):
        self.current_position = position.copy()
        self.current_rotation = rotation.copy()
        self.root_acceleration = (velocity - self.current_velocity) / self.dt # this ignores smoothing
        self.current_velocity = velocity.copy()
        self.current_angular_velocity = angular_velocity.copy()    
    
    def update_camera_state(self, azimuth: float, altitude: float, distance: float):
        self.camera_azimuth = azimuth
        self.camera_altitude = altitude
        self.camera_distance = distance
    
    def to_runtime_controls_kwargs(self) -> dict:
        return {
            'control_type': self.control_type,
            'gamepad_stick_left': self.gamepad_stick_left,
            'gamepad_stick_right': self.gamepad_stick_right,
            'camera_azimuth': self.camera_azimuth,
            'desired_strafe': self.desired_strafe,
            'current_rotation': self.current_rotation,
            'current_position': self.current_position,
            'current_velocity': self.current_velocity,
            'current_angular_velocity': self.current_angular_velocity,
            'movement_speed': self.movement_speed,
            'velocity_halflife': self.velocity_halflife,
            'rotation_halflife': self.rotation_halflife,
            'dt': self.dt,
        }
    
    def draw_debug_visuals(self):
        """Draw debug visualizations based on current control mode"""
        
        position = self.current_position

        if self.control_type == 'velocity_facing':
            velocity = self.current_velocity
            magnitude = np.linalg.norm(velocity)
            if magnitude > 0.05:
                start = position + np.array([0, 0.1, 0])
                end = start + velocity * 0.2
                DrawCapsule(Vector3(*start), Vector3(*end), 0.01, 5, 7, COLOR_VELOCITY)

                direction = velocity / magnitude
                if abs(direction[1]) > 0.99:
                    perp = np.array([1, 0, 0])
                else:
                    perp = np.cross(direction, np.array([0, 1, 0]))
                    perp = perp / np.linalg.norm(perp)

                head_size = 0.1
                left = end - direction * head_size + perp * head_size * 0.5
                right = end - direction * head_size - perp * head_size * 0.5
                
                DrawCapsule(Vector3(*end), Vector3(*left), 0.01, 5, 7, COLOR_VELOCITY)
                DrawCapsule(Vector3(*end), Vector3(*right), 0.01, 5, 7, COLOR_VELOCITY)

            facing_mag = np.linalg.norm(self.facing_direction)
            if facing_mag > 0.01:
                facing_pos = position
                facing_dir = self.facing_direction / facing_mag
                DrawSphere(Vector3(*facing_pos), 0.05, COLOR_FACING)
                DrawCapsule(Vector3(*facing_pos), Vector3(*(facing_pos + 0.25 * facing_dir)), 0.01, 5, 7, COLOR_FACING)
        
        # Trajectory path for trajectory mode
        if self.control_type == 'trajectory':
            for i in range(len(self.trajectory_positions)):
                pos = self.trajectory_positions[i]
                dir = self.trajectory_directions[i]
                DrawSphere(Vector3(*pos), 0.05, COLOR_TRAJECTORY)
                DrawCapsule(Vector3(*pos), Vector3(*(pos + 0.25 * dir)), 0.01, 5, 7, COLOR_TRAJECTORY)
    
    def draw_joystick_debug(self, gamepad_id=0, texture=None, screen_width=1280, screen_height=720):
        if IsGamepadAvailable(gamepad_id):
            left_stick_x = float(self.gamepad_stick_left[0])
            left_stick_y = float(self.gamepad_stick_left[1])
            right_stick_x = float(self.gamepad_stick_right[0])
            right_stick_y = float(self.gamepad_stick_right[1])

            scale = 0.5
            
            if texture is not None:
                texture_width = texture.width * scale
                texture_height = texture.height * scale
                pos_x = screen_width - texture_width - 10
                pos_y = screen_height - texture_height - 10

                DrawTextureEx(texture, Vector2(pos_x, pos_y), 0.0, scale, DARKGRAY)
                
                left_stick_center_x = pos_x + 259 * scale
                left_stick_center_y = pos_y + 152 * scale
                
                DrawCircle(int(left_stick_center_x), int(left_stick_center_y), 39 * scale, BLACK)
                DrawCircle(int(left_stick_center_x), int(left_stick_center_y), 34 * scale, LIGHTGRAY)
                DrawCircle(
                    int(left_stick_center_x + left_stick_x * 20 * scale),
                    int(left_stick_center_y + left_stick_y * 20 * scale),
                    25 * scale,
                    BLACK
                )
                
                right_stick_center_x = pos_x + 461 * scale
                right_stick_center_y = pos_y + 237 * scale
                
                DrawCircle(int(right_stick_center_x), int(right_stick_center_y), 38 * scale, BLACK)
                DrawCircle(int(right_stick_center_x), int(right_stick_center_y), 33 * scale, LIGHTGRAY)
                DrawCircle(
                    int(right_stick_center_x + right_stick_x * 20 * scale),
                    int(right_stick_center_y + right_stick_y * 20 * scale),
                    25 * scale,
                    BLACK
                )

    def __repr__(self) -> str:
        return (
            f"GameplayInput(\n"
            f"  control_type={self.control_type},\n"
            f"  stick_left=[{self.gamepad_stick_left[0]:.2f}, {self.gamepad_stick_left[1]:.2f}],\n"
            f"  stick_right=[{self.gamepad_stick_right[0]:.2f}, {self.gamepad_stick_right[1]:.2f}],\n"
            f"  position=[{self.current_position[0]:.2f}, {self.current_position[1]:.2f}, {self.current_position[2]:.2f}],\n"
            f"  velocity=[{self.current_velocity[0]:.2f}, {self.current_velocity[1]:.2f}, {self.current_velocity[2]:.2f}],\n"
            f"  camera_azimuth={np.degrees(self.camera_azimuth):.1f}°\n"
            f")"
        )
