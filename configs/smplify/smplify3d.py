type = 'SMPLify'
verbose = True

body_model = dict(
    type='SMPL',
    gender='neutral',
    num_betas=10,
    keypoint_src='smpl_45',
    keypoint_dst='smpl_45',
    # keypoint_dst='smpl',
    model_path='data/body_models/smpl',
    batch_size=1)

stages = [
    # stage 0
    dict(
        warmup=True,
        num_iter=10,
        fit_global_orient=True,
        fit_transl=True,
        fit_body_pose=False,
        fit_betas=False,
        joint_weights=dict(
            body_weight=5.0,
            use_shoulder_hip_only=True,
        )),
    # stage 1
    dict(
        num_iter=1,
        fit_global_orient=True,
        fit_transl=True,
        fit_body_pose=False,
        fit_betas=False,
        joint_weights=dict(
            body_weight=5.0,
            use_shoulder_hip_only=True,
        )),
    # stage 2
    dict(
        num_iter=1,
        fit_global_orient=True,
        fit_transl=True,
        fit_body_pose=True,
        fit_betas=True,
        joint_weights=dict(body_weight=5.0, use_shoulder_hip_only=False))
]

optimizer = dict(
    type='LBFGS', max_iter=30, lr=1.0, line_search_fn='strong_wolfe')

keypoints3d_loss = dict(
    type='KeypointMSELoss', loss_weight=10.0, reduction='sum', sigma=100)

shape_prior_loss = dict(type='ShapePriorLoss', loss_weight=1, reduction='sum')

smooth_loss = dict(type='SmoothJointLoss', loss_weight=1.0, reduction='sum')

# pose_prior_loss = dict(
#     type='MaxMixturePrior',
#     prior_folder='data',
#     num_gaussians=8,
#     loss_weight=4.78**2,
#     reduction='sum')

ignore_keypoints = [
    'neck_openpose', 'right_hip_openpose', 'left_hip_openpose',
    'right_hip_extra', 'left_hip_extra'
]
