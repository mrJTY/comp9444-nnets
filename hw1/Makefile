
.venv:
	conda create -n torch python=3.6

install:
	conda install pytorch torchvision cpuonly -c pytorch && conda install tqdm matplotlib

submit:
	cd src && give cs9444 hw1 part1.py part2.py part3.py

check:
	9444 classrun -check
