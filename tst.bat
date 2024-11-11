REM python makemore.py --device=cpu --gentext=1 --block_size=32 --learning-rate=0.001
python makemore.py --device=cuda --gentext=1 --block_size=32 --learning-rate=0.001
python makemore.py --device=cuda --gentext=1 --block_size=32 --learning-rate=0.001 --max-steps=10000
REM python makemore.py --device=cpu --gentext=1 --block_size=32 --learning-rate=0.001 --max-steps=10000
