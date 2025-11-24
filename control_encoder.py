from typing import Any, List, Dict, Tuple
from abc import ABC, abstractmethod
import numpy as np
import control_operators as co
import torch
import torch.nn as nn
import quat
from gameplay_input import GameplayInput
from train import intersect_tagged_ranges 


class ControlEncoderBase(nn.Module, ABC):
    """
    Base class for a Control Encoder.
    
    This presents a similar interface to what designers and technical animators can implement in Blueprints in 
    the Unreal Engine implementation described in the paper.
    
    Must Implement:
        (1) Control Schema
        (2) Training Controls
        (3) Runtime Controls
    """
    def __init__(self) -> None:
        super().__init__()
        self.root: co.ControlOperator = self.build_control_schema()
    
    @abstractmethod
    def build_control_schema(self) -> co.ControlOperator:
        raise NotImplementedError
    
    @abstractmethod
    def training_controls(self, pose_data : Dict) -> Tuple[np.ndarray,List[Any]]:
        """ 
        Generate possible training control inputs from dataset pose_data
        returns a batch of pose indices and a list of controls List[Any] matching self.root operator contract
        """
        raise NotImplementedError

    @abstractmethod
    def runtime_controls(self, gameplay_input: GameplayInput) -> Any:
        raise NotImplementedError

    def output_size(self) -> int:
        return self.root.output_size()
    
    def forward(self, v: Any) -> torch.FloatTensor:
        return self.root(v)


class NullControlEncoder(ControlEncoderBase):
    """
    Control Schema: Null() -> R^0
    """
    def build_control_schema(self):
        return co.Null()
    
    def training_controls(self, pose_data : Dict):
        
        # Get range data
        range_starts = pose_data['range_starts']
        range_stops = pose_data['range_stops']
        range_lens = pose_data['range_lens']
        
        # Skip first frame in ranges to allow for validly indexing previous frame in training
        I = np.hstack([np.arange(rs + 1, re) for rs, re in zip(range_starts, range_stops)])
        V = [None] * len(I)
        
        return I, V
    
    def runtime_controls(self, gameplay_input: GameplayInput):
        return ('uncontrolled', None)



class UberControlEncoder(ControlEncoderBase):
    
    def __init__(self) -> None:
        super().__init__()

        # velocity
        self.lookahead_time = 0.25 

        # trajectory
        self.future_frames = np.asarray([20, 40, 60], dtype=np.int32)

    def build_control_schema(self) -> co.ControlOperator:
        return co.Or({
            "uncontrolled": co.Null(),
            "velocity_facing": co.And({
                "velocity": co.Velocity(),
                "direction": co.Optional(co.Direction())
            }),
            "trajectory": co.FixedArray(
                co.And({
                    "location": co.Location(),
                    "direction": co.Direction(),
                }),
                3
            )
        })

    def training_controls(self, pose_data : Dict) -> Tuple[np.ndarray,List[Any]]:

        # Get data from the pose database
        
        range_starts = pose_data['range_starts']
        range_stops = pose_data['range_stops']
        range_names = pose_data['range_names']
        tag_range_starts = pose_data['tag_range_starts']
        tag_range_stops = pose_data['tag_range_stops']
        tag_tags = pose_data['tag_tags']
        Xroot_pos = pose_data['Xpos'][:,0]
        Xroot_rot = pose_data['Xrot'][:,0]
        Xroot_vel = pose_data['Xvel'][:,0]
        Xroot_dir = quat.mul_vec(Xroot_rot, np.array([0,0,1]))
        
        # Get locomotion tag ranges
        # use intersect_tagged_ranges for intersection:
        #   locomotion_range_starts, locomotion_range_stops = intersect_tagged_ranges(
        #       tag_range_starts, tag_range_stops, tag_tags, ['locomotion', 'style1'])
        locomotion_range_starts, locomotion_range_stops = intersect_tagged_ranges(
            tag_range_starts, tag_range_stops, tag_tags, ['locomotion'])

        # Prepare all pose indices and control values

        I, V = [], []

        for control_type in ["uncontrolled", "trajectory", "velocity_facing"]:

            if control_type == "uncontrolled":    
                
                # uncontrolled is valid for all ranges
                for rs, re in zip(range_starts, range_stops):

                    # Get all the pose indices for the range
                    range_pose_indices = np.arange(rs + 1, re)

                    # Construct the control objects
                    I.append(range_pose_indices)
                    V.append([(control_type, None) for _ in range(len(range_pose_indices))])
                
            elif control_type == "trajectory":
                
                # Use locomotion tag ranges
                for rs, re in zip(locomotion_range_starts, locomotion_range_stops):
                    
                    # Get all the pose indices for the range
                    range_pose_indices = np.arange(rs + 1, re - np.max(self.future_frames))
                
                    # Compute the future local positions
                    future_pos = torch.as_tensor(quat.inv_mul_vec(Xroot_rot[range_pose_indices][:,None], Xroot_pos[range_pose_indices[:,None] + self.future_frames] - Xroot_pos[range_pose_indices][:,None]), dtype=torch.float32)
                    future_dir = torch.as_tensor(quat.inv_mul_vec(Xroot_rot[range_pose_indices][:,None], Xroot_dir[range_pose_indices[:,None] + self.future_frames]), dtype=torch.float32)
                    
                    # Construct the control objects
                    I.append(range_pose_indices)
                    V.append([(control_type, [{"location": cpos[f], "direction": cdir[f]} for f in range(len(self.future_frames))])
                        for cpos, cdir in zip(future_pos, future_dir)])
                    
            elif control_type == "velocity_facing":
                # Get the amount to look into the future
                shift = int(round(self.lookahead_time * 60.0))
                
                # Use locomotion tag ranges
                for rs, re in zip(locomotion_range_starts, locomotion_range_stops):
                    
                    # Get all the pose indices for the range
                    range_pose_indices = np.arange(rs + 1, re - shift)
                    
                    # Compute the future local velocity
                    future_vel = torch.as_tensor(quat.inv_mul_vec(Xroot_rot[range_pose_indices], Xroot_vel[range_pose_indices + shift]), dtype=torch.float32)

                    # Compute the future looking direction
                    future_dir = torch.as_tensor(quat.inv_mul_vec(Xroot_rot[range_pose_indices], Xroot_dir[range_pose_indices + shift]), dtype=torch.float32)

                    # Generate samples with and without facing direction
                    for direction in [None, future_dir]:
                        I.append(range_pose_indices)
                        V.append([
                            (control_type, {
                                "velocity": cvel,
                                "direction": cdir if direction is not None else None
                            })
                            for cvel, cdir in zip(future_vel, direction if direction is not None else [None] * len(future_vel))
                        ])

        # Concatenate Together
        I, V = np.concatenate(I, axis=0), sum(V, [])
        
        assert len(I) == len(V)
        
        return I, V

    def runtime_controls(self, gameplay_input: GameplayInput) -> Any:
        
        params = gameplay_input.to_runtime_controls_kwargs()
        control_type = params['control_type']
        current_rotation = params['current_rotation']
        current_position = params['current_position']
        current_velocity = params['current_velocity']
        current_angular_velocity = params['current_angular_velocity']
        velocity_halflife = params['velocity_halflife']
        rotation_halflife = params['rotation_halflife']
        
        # For uncontroller simply give None
        if control_type == 'uncontrolled':
            return (control_type, None)
        
        # For trajectory control predict trajectory from gamepad
        elif control_type == 'trajectory':
            
            # Spring Trajectory Methods taken from here:
            # https://theorangeduck.com/page/spring-roll-call#controllers
            
            def halflife_to_damping(halflife, eps=1e-5):
                return (4.0 * 0.69314718056) / (halflife + eps)


            def trajectory_spring_position(pos, vel, acc, desired_vel, halflife, dt):
                y = halflife_to_damping(halflife) / 2.0
                j0 = vel - desired_vel
                j1 = acc + j0 * y
                eydt = np.exp(-y * dt)
                
                return (
                    eydt * (((-j1) / (y * y)) + ((-j0 - j1 * dt) / y)) + (j1 / (y * y)) + j0 / y + desired_vel * dt + pos,
                    eydt * (j0 + j1 * dt) + desired_vel,
                    eydt * (acc - j1 * y * dt)
                )


            def trajectory_spring_rotation(rot, ang, desired_rot, halflife, dt):
                y = halflife_to_damping(halflife) / 2.0
                j0 = quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rot, desired_rot)))
                j1 = ang + j0 * y
                eydt = np.exp(-y * dt)
                
                return (
                    quat.mul(quat.from_scaled_angle_axis(eydt * (j0 + j1 * dt)), desired_rot),
                    eydt * (ang - j1 * y * dt)
                )
            
            # Get the stick directions
            global_stick_direction = gameplay_input.movement_direction_world
            global_facing_direction = gameplay_input.facing_direction_world
            
            # Compute desired velocity
            desired_vel = global_stick_direction * gameplay_input.effective_movement_speed
            
            # Determine desired direction from right stick or movement
            if global_facing_direction is not None:
                desired_dir = global_facing_direction
            elif gameplay_input.left_stick_magnitude > 0.01:
                desired_dir = global_stick_direction / np.linalg.norm(global_stick_direction)
            else:
                desired_dir = quat.mul_vec(current_rotation, np.array([0, 0, 1]))
            
            desired_rot = quat.normalize(quat.between(np.array([0, 0, 1]), desired_dir))
            
            # Get current trajectory state from gameplay_input
            root_pos = current_position
            root_vel = current_velocity
            root_acc = gameplay_input.root_acceleration
            root_rot = current_rotation
            root_ang = current_angular_velocity
            
            # Predict trajectory using spring dynamics
            Ttimes = np.array([20, 40, 60]) / 60.0
            
            Tpos, _, _ = trajectory_spring_position(
                root_pos, root_vel, root_acc, desired_vel, velocity_halflife, Ttimes[..., None])
            
            Trot, _ = trajectory_spring_rotation(
                root_rot, root_ang, desired_rot, rotation_halflife, Ttimes[..., None])
            
            Tdir = quat.mul_vec(Trot, np.array([0, 0, 1]))
            
            # Update gameplay_input trajectory state for visualization
            gameplay_input.trajectory_positions = Tpos.copy()
            gameplay_input.trajectory_directions = Tdir.copy()
            
            # Convert to local coordinates
            local_traj_pos = quat.inv_mul_vec(current_rotation, Tpos - root_pos)
            local_traj_dir = quat.inv_mul_vec(current_rotation, Tdir)
            
            # Build trajectory control list matching training format
            trajectory_list = [
                {"location": torch.as_tensor(local_traj_pos[i], dtype=torch.float32), 
                 "direction": torch.as_tensor(local_traj_dir[i], dtype=torch.float32)}
                for i in range(len(Ttimes))
            ]
            
            return ('trajectory', trajectory_list)
        
        # For velocity facing use left stick for velocity, and right stick for facing (when it is given)
        elif control_type == "velocity_facing":
            
            # Get current stick directions
            global_stick_direction = gameplay_input.movement_direction_world
            local_stick_direction = quat.inv_mul_vec(current_rotation, global_stick_direction)
            local_desired_velocity = local_stick_direction * gameplay_input.effective_movement_speed
            global_facing_direction = gameplay_input.facing_direction_world
            
            # If facing direction is provided then compute it otherwise it will not be used
            if global_facing_direction is not None:
                local_desired_facing = quat.inv_mul_vec(current_rotation, global_facing_direction)
                local_desired_facing_tensor = torch.as_tensor(local_desired_facing, dtype=torch.float32)

                gameplay_input.facing_direction = global_facing_direction
            else:
                gameplay_input.facing_direction = np.zeros(3, dtype=np.float32)
                local_desired_facing_tensor = None

            return ('velocity_facing', {
                "velocity": torch.as_tensor(local_desired_velocity, dtype=torch.float32),
                "direction": local_desired_facing_tensor
            })

        else:
            raise ValueError(f"Unknown control type: {control_type}")