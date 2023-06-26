
_base_ = [
    '../_base_/models/bonai_loft_foa_r50_fpn_prompt_train.py',
    '../_base_/datasets/bonai_instance_prompt.py',
    '../_base_/schedules/schedule_2x_bonai.py', 
    '../_base_/default_runtime.py'
]
# load_from = 'work_dirs/loft_foa_r50_fpn_2x_bonai/latest.pth'
# resume_from = 'work_dirs/loft_foa_r50_fpn_2x_bonai_prompt_train_g4/epoch_8.pth'