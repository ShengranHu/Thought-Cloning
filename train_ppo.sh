python scripts/train_ppo.py --env OpenDoorLoc --device 0 --env_num 256 --ppo_updates 50 --model_name model_open_door_seed1 --seed 1
python scripts/train_ppo.py --env OpenDoorLoc --device 0 --env_num 256 --ppo_updates 50 --model_name model_open_door_bc0_warmstart_seed1 --model bc_warmstart_door/bc0/model.pt --seed 1
python scripts/train_ppo.py --env OpenDoorLoc --device 0 --env_num 256 --ppo_updates 50 --model_name model_open_door_bc10_warmstart_seed1 --model bc_warmstart_door/bc10/model.pt --seed 1

python scripts/train_ppo.py --env UnlockPickupDist --device 0 --env_num 256 --ppo_updates 50 --model_name model_unlock_pickup_dist_seed1 --seed 1
python scripts/train_ppo.py --env UnlockPickupDist --device 0 --env_num 256 --ppo_updates 50 --model_name model_unlock_pickup_dist_bc0_warmstart_seed1 --model bc_warmstart_upd/bc0/model.pt --seed 1
python scripts/train_ppo.py --env UnlockPickupDist --device 0 --env_num 256 --ppo_updates 50 --model_name model_unlock_pickup_dist_bc10_warmstart_seed1 --model bc_warmstart_upd/bc10/model.pt --seed 1