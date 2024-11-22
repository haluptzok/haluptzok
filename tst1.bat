python makemore.py --device=cuda --gentext=1 --block_size=32 --learning-rate=0.001
python makemore.py --device=cuda --gentext=1 --block_size=32 --learning-rate=0.001 --max-steps=10001
python gpt_dev.py
