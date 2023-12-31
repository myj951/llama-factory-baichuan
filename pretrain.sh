accelerate launch src/train_bash.py \
--stage pt \
--model_name_or_path /backup/baichuan/Baichuan2-13B-Chat \
--do_train \
--dataset my_text \
--template default \
--finetuning_type full \
--output_dir /backup/Baichuan2-13B-Chat-pt-912 \
--overwrite_cache \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 2 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--learning_rate 1e-5 \
--num_train_epochs 160.0 \
--plot_loss \
--fp16 \
--template baichuan \
--max_source_length 1024 \
--max_target_length 1024 \
--overwrite_output_dir \
--val_size 10 \
--eval_steps 50
