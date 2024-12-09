# dataset selection:
# set data_ind (0, 1, 2, 3, 4) to select dataset in data_dir_list
# testing checkpoint
# --resume_checkpoint logs/2023-12-15-21-52-50-mpi3d_real_complex_FDAE_seed0_rank0/model100000.pt --eval_only True
data_ind=3
data_dir_list=(datasets/shapes3d datasets/cars3d datasets/mpi3d_toy datasets/celeba datasets/animals)
image_size_input_list=(64 64 64 224 224)
image_size_gen_list=(64 64 64 64 64)
group_num_list=(6 2 7 15 10)
code_dim_list=(80 80 100 100 80)
batch_size_list=(64 64 64 32 32)
encoder_type_list=("resnet18" "resnet18" "resnet18" "resnet18" "resnet18")
content_decorrelation_weight_list=(2.5e-5 2.5e-5 2.5e-5 8.5e-6 2e-5)
mask_entropy_weight_list=(3.5e-4 3.5e-4 3.5e-4 5e-6 2e-4)
data_dir=${data_dir_list[${data_ind}]}
image_size_input=${image_size_input_list[${data_ind}]}
image_size_gen=${image_size_gen_list[${data_ind}]}
group_num=${group_num_list[${data_ind}]}
code_dim=${code_dim_list[${data_ind}]}
batch_size=${batch_size_list[${data_ind}]}
content_decorrelation_weight=${content_decorrelation_weight_list[${data_ind}]} # originally 2.5e-5
mask_entropy_weight=${mask_entropy_weight_list[${data_ind}]} # originally 1.0e-4
encoder_type=${encoder_type_list[${data_ind}]}

learning_rate=1e-5
max_step=500_000
eval_interval=500_000
save_interval=2_000
for seed in 0
do
python fdae_train.py --log_suffix FDAE_seed${seed}_ \
--data_dir ${data_dir} \
--mask_entropy_weight ${mask_entropy_weight} --content_decorrelation_weight ${content_decorrelation_weight} \
--semantic_group_num ${group_num} --semantic_code_dim ${code_dim} --mask_code_dim ${code_dim} --semantic_code_adjust_dim ${code_dim} \
--additional_cond_map_layer_dim ${code_dim} \
--max_step ${max_step} --save_interval ${save_interval} --eval_interval ${eval_interval} --attention_resolutions 32,16,8 \
--global_batch_size ${batch_size} --lr ${learning_rate} \
--class_cond False --image_cond True --use_scale_shift_norm False --dropout 0.1 \
--image_size_input ${image_size_input} --image_size_gen ${image_size_gen} \
--num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --schedule_sampler lognormal \
--use_fp16 True --weight_decay 0.0 --weight_schedule karras --debug_mode True \
--seed ${seed} \
--encoder_type ${encoder_type}
done
