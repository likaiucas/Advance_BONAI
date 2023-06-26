_base_ = [
    '../_base_/models/bonai_loft_foa_r50_fpn_basic.py',
    '../_base_/datasets/bonai_instance.py',
    '../_base_/schedules/schedule_2x_bonai.py', 
    '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='LOFT',
    roi_head=dict(
        offset_head=dict(
            offset_coordinate='polar',
            offset_coder=dict(
                    type='DeltaPolarOffsetCoder',
                    target_means=[0.0, 0.0],
                    target_stds=[0.5, 0.5]),)))
# model training and testing settings

        
data = dict(
    train=dict(
        bbox_type='building',
        mask_type='roof',
        offset_coordinate='polar',
        ),
    test=dict(
        bbox_type='building',
        mask_type='roof',
        offset_coordinate='polar',
        ),
    val=dict(
        bbox_type='building',
        mask_type='roof',
        offset_coordinate='polar',
        ))