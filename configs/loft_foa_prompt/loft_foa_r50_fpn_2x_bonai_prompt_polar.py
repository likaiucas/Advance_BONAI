
_base_ = [
    '../_base_/models/bonai_loft_foa_r50_fpn_prompt.py',
    '../_base_/datasets/bonai_instance_prompt.py',
    '../_base_/schedules/schedule_2x_bonai.py', 
    '../_base_/default_runtime.py'
]
# model settings
model = dict(
    roi_head=dict(
        offset_head=dict(
            offset_coordinate='polar',
            offset_coder=dict(
                    type='DeltaPolarOffsetCoder',
                    target_means=[0.0, 0.0],
                    target_stds=[0.5, 0.5]),)))