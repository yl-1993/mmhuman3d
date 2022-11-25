from typing import Optional

import numpy as np
import torch
from smplx import SMPLH as _SMPLH
from smplx import SMPLHLayer as _SMPLHLayer
from smplx.lbs import vertices2joints

from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_num,
)
from mmhuman3d.core.conventions.segmentation import body_segmentation


class SMPLH(_SMPLH):
    """Extension of the official SMPL-X implementation."""

    body_pose_keys = {'global_orient', 'body_pose'}
    full_pose_keys = {
        'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose'
    }
    NUM_VERTS = 6890
    NUM_FACES = 13776

    def __init__(self,
                 *args,
                 keypoint_src: str = 'smplh',
                 keypoint_dst: str = 'human_data',
                 keypoint_approximate: bool = False,
                 joints_regressor: str = None,
                 extra_joints_regressor: str = None,
                 **kwargs):
        """
        Args:
            *args: extra arguments for SMPL initialization.
            keypoint_src: source convention of keypoints. This convention
                is used for keypoints obtained from joint regressors.
                Keypoints then undergo conversion into keypoint_dst
                convention.
            keypoint_dst: destination convention of keypoints. This convention
                is used for keypoints in the output.
            keypoint_approximate: whether to use approximate matching in
                convention conversion for keypoints.
            joints_regressor: path to joint regressor. Should be a .npy
                file. If provided, replaces the official J_regressor of SMPL.
            extra_joints_regressor: path to extra joint regressor. Should be
                a .npy file. If provided, extra joints are regressed and
                concatenated after the joints regressed with the official
                J_regressor or joints_regressor.
            **kwargs: extra keyword arguments for SMPL initialization.

        Returns:
            None
        """
        super(SMPLH, self).__init__(*args, **kwargs)
        # joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        self.keypoint_src = keypoint_src
        self.keypoint_dst = keypoint_dst
        self.keypoint_approximate = keypoint_approximate

        # override the default SMPL joint regressor if available
        if joints_regressor is not None:
            joints_regressor = torch.tensor(
                np.load(joints_regressor), dtype=torch.float)
            self.register_buffer('joints_regressor', joints_regressor)

        # allow for extra joints to be regressed if available
        if extra_joints_regressor is not None:
            joints_regressor_extra = torch.tensor(
                np.load(extra_joints_regressor), dtype=torch.float)
            self.register_buffer('joints_regressor_extra',
                                 joints_regressor_extra)

        self.num_verts = self.get_num_verts()
        self.num_joints = get_keypoint_num(convention=self.keypoint_dst)
        self.body_part_segmentation = body_segmentation('smplx')

    def forward(self,
                *args,
                return_verts: bool = True,
                return_full_pose: bool = False,
                **kwargs) -> dict:
        """Forward function.

        Args:
            *args: extra arguments for SMPL
            return_verts: whether to return vertices
            return_full_pose: whether to return full pose parameters
            **kwargs: extra arguments for SMPL

        Returns:
            output: contains output parameters and attributes
        """

        kwargs['get_skin'] = True
        smplx_output = super(SMPLH, self).forward(*args, **kwargs)

        if not hasattr(self, 'joints_regressor'):
            joints = smplx_output.joints
        else:
            joints = vertices2joints(self.joints_regressor,
                                     smplx_output.vertices)

        if hasattr(self, 'joints_regressor_extra'):
            extra_joints = vertices2joints(self.joints_regressor_extra,
                                           smplx_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)

        joints, joint_mask = convert_kps(
            joints,
            src=self.keypoint_src,
            dst=self.keypoint_dst,
            approximate=self.keypoint_approximate)
        if isinstance(joint_mask, np.ndarray):
            joint_mask = torch.tensor(
                joint_mask, dtype=torch.uint8, device=joints.device)

        batch_size = joints.shape[0]
        joint_mask = joint_mask.reshape(1, -1).expand(batch_size, -1)

        output = dict(
            global_orient=smplx_output.global_orient,
            body_pose=smplx_output.body_pose,
            joints=joints,
            joint_mask=joint_mask,
            keypoints=torch.cat([joints, joint_mask[:, :, None]], dim=-1),
            betas=smplx_output.betas)

        if return_verts:
            output['vertices'] = smplx_output.vertices
        if return_full_pose:
            output['full_pose'] = smplx_output.full_pose

        return output

    @classmethod
    def tensor2dict(cls,
                    full_pose: torch.Tensor,
                    betas: Optional[torch.Tensor] = None,
                    transl: Optional[torch.Tensor] = None) -> dict:
        """Convert full pose tensor to pose dict.

        Args:
            full_pose (torch.Tensor): shape should be (..., 165) or
                (..., 55, 3). All zeros for T-pose.
            betas (Optional[torch.Tensor], optional): shape should be
                (..., 10). The batch num should be 1 or corresponds with
                full_pose.
                Defaults to None.
            transl (Optional[torch.Tensor], optional): shape should be
                (..., 3). The batch num should be 1 or corresponds with
                full_pose.
                Defaults to None.

        Returns:
            dict: dict of smplx pose containing transl & betas.
        """
        NUM_BODY_JOINTS = cls.NUM_BODY_JOINTS
        NUM_HAND_JOINTS = cls.NUM_HAND_JOINTS
        NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS
        full_pose = full_pose.view(-1, (NUM_JOINTS + 1), 3)
        global_orient = full_pose[:, :1]
        body_pose = full_pose[:, 1:NUM_BODY_JOINTS + 1]
        print(body_pose.shape, NUM_BODY_JOINTS, NUM_HAND_JOINTS)
        left_hand_pose = full_pose[:, NUM_BODY_JOINTS + 1:NUM_BODY_JOINTS + 16]
        right_hand_pose = full_pose[:,
                                    NUM_BODY_JOINTS + 16:NUM_BODY_JOINTS + 31]
        batch_size = body_pose.shape[0]
        if betas is not None:
            # squeeze or unsqueeze betas to 2 dims
            betas = betas.view(-1, betas.shape[-1])
            if betas.shape[0] == 1:
                betas = betas.repeat(batch_size, 1)
        else:
            betas = betas
        transl = transl.view(batch_size, -1) if transl is not None else transl
        return {
            'betas':
            betas,
            'global_orient':
            global_orient.view(batch_size, 3),
            'body_pose':
            body_pose.view(batch_size, NUM_BODY_JOINTS * 3),
            'left_hand_pose':
            left_hand_pose.view(batch_size, NUM_HAND_JOINTS * 3),
            'right_hand_pose':
            right_hand_pose.view(batch_size, NUM_HAND_JOINTS * 3),
            'transl':
            transl,
        }

    @classmethod
    def dict2tensor(cls, smplx_dict: dict) -> torch.Tensor:
        """Convert smplx pose dict to full pose tensor.

        Args:
            smplx_dict (dict): smplx pose dict.

        Returns:
            torch: full pose tensor.
        """
        assert cls.body_pose_keys.issubset(smplx_dict)
        for k in smplx_dict:
            if isinstance(smplx_dict[k], np.ndarray):
                smplx_dict[k] = torch.Tensor(smplx_dict[k])
        NUM_BODY_JOINTS = cls.NUM_BODY_JOINTS
        NUM_HAND_JOINTS = cls.NUM_HAND_JOINTS
        NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS
        global_orient = smplx_dict['global_orient'].reshape(-1, 1, 3)
        body_pose = smplx_dict['body_pose'].reshape(-1, NUM_BODY_JOINTS, 3)
        batch_size = global_orient.shape[0]
        left_hand_pose = smplx_dict.get(
            'left_hand_pose', torch.zeros((batch_size, NUM_HAND_JOINTS, 3)))
        right_hand_pose = smplx_dict.get(
            'right_hand_pose', torch.zeros((batch_size, NUM_HAND_JOINTS, 3)))
        full_pose = torch.cat([
            global_orient, body_pose,
            left_hand_pose.reshape(-1, 15, 3),
            right_hand_pose.reshape(-1, 15, 3)
        ],
                              dim=1).reshape(-1, (NUM_JOINTS + 1) * 3)
        return full_pose


class SMPLHLayer(_SMPLHLayer):
    """Extension of the official SMPL-X implementation."""

    body_pose_keys = {'global_orient', 'body_pose'}
    full_pose_keys = {
        'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose',
    }
    NUM_VERTS = 10475
    NUM_FACES = 20908

    def __init__(self,
                 *args,
                 keypoint_src: str = 'smplx',
                 keypoint_dst: str = 'human_data',
                 keypoint_approximate: bool = False,
                 joints_regressor: str = None,
                 extra_joints_regressor: str = None,
                 **kwargs):
        """
        Args:
            *args: extra arguments for SMPL initialization.
            keypoint_src: source convention of keypoints. This convention
                is used for keypoints obtained from joint regressors.
                Keypoints then undergo conversion into keypoint_dst
                convention.
            keypoint_dst: destination convention of keypoints. This convention
                is used for keypoints in the output.
            keypoint_approximate: whether to use approximate matching in
                convention conversion for keypoints.
            joints_regressor: path to joint regressor. Should be a .npy
                file. If provided, replaces the official J_regressor of SMPL.
            extra_joints_regressor: path to extra joint regressor. Should be
                a .npy file. If provided, extra joints are regressed and
                concatenated after the joints regressed with the official
                J_regressor or joints_regressor.
            **kwargs: extra keyword arguments for SMPL initialization.

        Returns:
            None
        """
        super(SMPLHLayer, self).__init__(*args, **kwargs)
        # joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        self.keypoint_src = keypoint_src
        self.keypoint_dst = keypoint_dst
        self.keypoint_approximate = keypoint_approximate

        # override the default SMPL joint regressor if available
        if joints_regressor is not None:
            joints_regressor = torch.tensor(
                np.load(joints_regressor), dtype=torch.float)
            self.register_buffer('joints_regressor', joints_regressor)

        # allow for extra joints to be regressed if available
        if extra_joints_regressor is not None:
            joints_regressor_extra = torch.tensor(
                np.load(extra_joints_regressor), dtype=torch.float)
            self.register_buffer('joints_regressor_extra',
                                 joints_regressor_extra)

        self.num_verts = self.get_num_verts()
        self.num_joints = get_keypoint_num(convention=self.keypoint_dst)
        self.body_part_segmentation = body_segmentation('smplx')

    def forward(self,
                *args,
                return_verts: bool = True,
                return_full_pose: bool = False,
                **kwargs) -> dict:
        """Forward function.

        Args:
            *args: extra arguments for SMPL
            return_verts: whether to return vertices
            return_full_pose: whether to return full pose parameters
            **kwargs: extra arguments for SMPL

        Returns:
            output: contains output parameters and attributes
        """

        kwargs['get_skin'] = True
        smplx_output = super(SMPLHLayer, self).forward(*args, **kwargs)

        if not hasattr(self, 'joints_regressor'):
            joints = smplx_output.joints
        else:
            joints = vertices2joints(self.joints_regressor,
                                     smplx_output.vertices)

        if hasattr(self, 'joints_regressor_extra'):
            extra_joints = vertices2joints(self.joints_regressor_extra,
                                           smplx_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)

        joints, joint_mask = convert_kps(
            joints,
            src=self.keypoint_src,
            dst=self.keypoint_dst,
            approximate=self.keypoint_approximate)
        if isinstance(joint_mask, np.ndarray):
            joint_mask = torch.tensor(
                joint_mask, dtype=torch.uint8, device=joints.device)

        batch_size = joints.shape[0]
        joint_mask = joint_mask.reshape(1, -1).expand(batch_size, -1)

        output = dict(
            global_orient=smplx_output.global_orient,
            body_pose=smplx_output.body_pose,
            joints=joints,
            joint_mask=joint_mask,
            keypoints=torch.cat([joints, joint_mask[:, :, None]], dim=-1),
            betas=smplx_output.betas)

        if return_verts:
            output['vertices'] = smplx_output.vertices
        if return_full_pose:
            output['full_pose'] = smplx_output.full_pose

        return output
