# save masks for images in --data_dir
# for mpi3d or shapes3d
# python conditional_image_sample.py \
# --model_path logs/2024-09-04-12-16-03-shapes3d_FDAE_seed0_rank0/model010000.pt \
# --seed 0 \
# --num_samples 5 \
# --batch_size 5 \
# --data_dir visualization/shapes3d_test \
# --save_mask True \
# --class_cond False \
# --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --weight_schedule karras \
# --attention_resolutions 32,16,8 --use_scale_shift_norm False --dropout 0.0 --image_size 64 \
# --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_fp16 True

# for celeba
python conditional_image_sample.py \
--model_path logs/2024-11-26-12-29-24-celeba_FDAE_seed0_rank0/model076000.pt \
--seed 0 \
--num_samples 5 \
--batch_size 5 \
--data_dir visualization/celeba_test \
--save_mask True \
--class_cond False \
--sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --weight_schedule karras \
--attention_resolutions 32,16,8 --use_scale_shift_norm False --dropout 0.0 --image_size_input 224 --image_size_gen 64 \
--num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_fp16 True \
# --use_image_input True 


# for 64X64 datasets
# python conditional_image_sample.py \
# --model_path logs/2024-09-12-13-45-05-mpi3d_toy_FDAE_seed0_rank0/model070000.pt \
# --seed 0 \
# --num_samples 5 \
# --batch_size 5 \
# --data_dir visualization/mpi3d_toy_test \
# --save_mask True \
# --class_cond False \
# --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --weight_schedule karras \
# --attention_resolutions 32,16,8 --use_scale_shift_norm False --dropout 0.0 --image_size 64 \
# --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_fp16 True

# swapping content codes or mask codes for each pair of images in --data_dir
# set swap_flag=True
# for factor_dim, c1,2-m0 denotes swapping content code 1 and 2, c0-m1,2 denotes swapping mask codes 1 and 2
# python conditional_image_sample.py \
# --model_path logs/2024-09-09-18-01-00-mpi3d_toy_FDAE_seed0_rank0/model010000.pt \
# --swap_flag True --factor_dim c1,2,3,4,5,6-m0 \
# --seed 0 \
# --num_samples 5 \
# --batch_size 5 \
# --data_dir visualization/mpi3d_toy_test \
# --class_cond False \
# --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --weight_schedule karras \
# --attention_resolutions 32,16,8 --use_scale_shift_norm False --dropout 0.0 --image_size 64 \
# --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_fp16 True 
