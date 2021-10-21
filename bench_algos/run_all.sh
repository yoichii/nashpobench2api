# run six algorithms
python random_search.py --test_epoch both --time_budget 20000 --save_dir logs/random &
python rea.py --test_epoch both --time_budget 20000 --save_dir logs/rea &
python reinforce.py --test_epoch both --time_budget 20000 --save_dir logs/reinforce &
python bohb.py --test_epoch both --time_budget 20000 --save_dir logs/bohb &
python rs_then_rea.py --test_epoch both --time_budget_hpo 5000 --time_budget 20000 --save_dir logs/rs_then_rea &
python bohb_then_rea.py --test_epoch both --time_budget 20000 --save_dir logs/bohb_then_rea &
