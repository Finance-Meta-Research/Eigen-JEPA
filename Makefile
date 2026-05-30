.PHONY: test smoke train benchmark suite paper clean

PYTHON ?= python

smoke:
	$(PYTHON) -m eigen_jepa.train --deterministic --device cpu --out_dir results/smoke --epochs 1 --num_train 16 --num_val 8 --num_test 8 --num_assets 6 --total_steps 80 --context_len 10 --horizon 4 --batch_size 4 --d_model 24 --latent_dim 24 --memory_size 16 --memory_top_k 2

train:
	$(PYTHON) -m eigen_jepa.train --deterministic --device cpu --out_dir results/train

benchmark:
	$(PYTHON) -m eigen_jepa.benchmark --deterministic --device cpu --out_dir results/benchmark --num_seeds 3

suite:
	$(PYTHON) -m eigen_jepa.suite --deterministic --device cpu --out_dir results/suite --num_seeds 2

paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf results/* paper/*.aux paper/*.log paper/*.out paper/_render* paper/*.toc
