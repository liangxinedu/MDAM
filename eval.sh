file=result.log
if [ "$1" = download ]
then
	python download_pretrained_models.py
	unzip pretrained.zip
fi

if [ "$1" = generate_data ]
then
	mkdir data
	python generate_data.py --filename=data/tsp_20 --problem="tsp" --graph_sizes=20 --dataset_size=100000
	python generate_data.py --filename=data/tsp_50 --problem="tsp" --graph_sizes=50 --dataset_size=100000
	python generate_data.py --filename=data/tsp_100 --problem="tsp" --graph_sizes=100 --dataset_size=100000
	python generate_data.py --filename=data/vrp_20 --problem="vrp" --graph_sizes=20 --dataset_size=100000
	python generate_data.py --filename=data/vrp_50 --problem="vrp" --graph_sizes=50 --dataset_size=100000
	python generate_data.py --filename=data/vrp_100 --problem="vrp" --graph_sizes=100 --dataset_size=100000
	python generate_data.py --filename=data/op_20 --problem="op" --graph_sizes=20 --dataset_size=100000 --data_distribution="dist"
	python generate_data.py --filename=data/op_50 --problem="op" --graph_sizes=50 --dataset_size=100000 --data_distribution="dist"
	python generate_data.py --filename=data/op_100 --problem="op" --graph_sizes=100 --dataset_size=100000 --data_distribution="dist"
	python generate_data.py --filename=data/pctsp_20 --problem="pctsp" --graph_sizes=20 --dataset_size=100000
	python generate_data.py --filename=data/pctsp_50 --problem="pctsp" --graph_sizes=50 --dataset_size=100000
	python generate_data.py --filename=data/pctsp_100 --problem="pctsp" --graph_sizes=100 --dataset_size=100000
fi

# ------------------------ tsp -----------------------
if [ "$1" = tsp ]
then
	echo "tsp beam_size = 1" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/tsp_20.pkl --model=pretrained/tsp_20/epoch-99.pt --beam_size=1 --eval_batch_size=1024 | tee -a $file
	echo "tsp beam_size = 30" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/tsp_20.pkl --model=pretrained/tsp_20/epoch-99.pt --beam_size=30 --eval_batch_size=1024 | tee -a $file
	echo "tsp beam_size = 50" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/tsp_20.pkl --model=pretrained/tsp_20/epoch-99.pt --beam_size=50 --eval_batch_size=1024 | tee -a $file

	echo "tsp beam_size = 1" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/tsp_50.pkl --model=pretrained/tsp_50/epoch-99.pt --beam_size=1 --eval_batch_size=1024 | tee -a $file
	echo "tsp beam_size = 30" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/tsp_50.pkl --model=pretrained/tsp_50/epoch-99.pt --beam_size=30 --eval_batch_size=1024 | tee -a $file
	echo "tsp beam_size = 50" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/tsp_50.pkl --model=pretrained/tsp_50/epoch-99.pt --beam_size=50 --eval_batch_size=512 | tee -a $file

	echo "tsp beam_size = 1" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/tsp_100.pkl --model=pretrained/tsp_100/epoch-99.pt --beam_size=1 --eval_batch_size=1024 | tee -a $file
	echo "tsp beam_size = 30" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/tsp_100.pkl --model=pretrained/tsp_100/epoch-99.pt --beam_size=30 --eval_batch_size=512 | tee -a $file
	echo "tsp beam_size = 50" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/tsp_100.pkl --model=pretrained/tsp_100/epoch-99.pt --beam_size=50 --eval_batch_size=256 | tee -a $file
fi
# ------------------------ cvrp -----------------------
if [ "$1" = cvrp ]
then
	echo "cvrp beam_size = 1" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/vrp_20.pkl --model=pretrained/vrp_20/epoch-99.pt --beam_size=1 --eval_batch_size=1024 | tee -a $file
	echo "cvrp beam_size = 30" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/vrp_20.pkl --model=pretrained/vrp_20/epoch-99.pt --beam_size=30 --eval_batch_size=1024 | tee -a $file
	echo "cvrp beam_size = 50" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/vrp_20.pkl --model=pretrained/vrp_20/epoch-99.pt --beam_size=50 --eval_batch_size=1024 | tee -a $file

	echo "cvrp beam_size = 1" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/vrp_50.pkl --model=pretrained/vrp_50/epoch-99.pt --beam_size=1 --eval_batch_size=1024 | tee -a $file
	echo "cvrp beam_size = 30" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/vrp_50.pkl --model=pretrained/vrp_50/epoch-99.pt --beam_size=30 --eval_batch_size=1024 | tee -a $file
	echo "cvrp beam_size = 50" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/vrp_50.pkl --model=pretrained/vrp_50/epoch-99.pt --beam_size=50 --eval_batch_size=512 | tee -a $file

	echo "cvrp beam_size = 1" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/vrp_100.pkl --model=pretrained/vrp_100/epoch-99.pt --beam_size=1 --eval_batch_size=1024 | tee -a $file
	echo "cvrp beam_size = 30" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/vrp_100.pkl --model=pretrained/vrp_100/epoch-99.pt --beam_size=30 --eval_batch_size=512 | tee -a $file
	echo "cvrp beam_size = 50" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/vrp_100.pkl --model=pretrained/vrp_100/epoch-99.pt --beam_size=50 --eval_batch_size=256 | tee -a $file
fi
# ------------------------ sdvrp -----------------------
if [ "$1" = sdvrp ]
then
	echo "sdvrp beam_size = 1" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/vrp_20.pkl --model=pretrained/sdvrp_20/epoch-99.pt --beam_size=1 --eval_batch_size=1024 | tee -a $file
	echo "sdvrp beam_size = 30" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/vrp_20.pkl --model=pretrained/sdvrp_20/epoch-99.pt --beam_size=30 --eval_batch_size=1024 | tee -a $file
	echo "sdvrp beam_size = 50" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/vrp_20.pkl --model=pretrained/sdvrp_20/epoch-99.pt --beam_size=50 --eval_batch_size=1024 | tee -a $file

	echo "sdvrp beam_size = 1" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/vrp_50.pkl --model=pretrained/sdvrp_50/epoch-99.pt --beam_size=1 --eval_batch_size=1024 | tee -a $file
	echo "sdvrp beam_size = 30" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/vrp_50.pkl --model=pretrained/sdvrp_50/epoch-99.pt --beam_size=30 --eval_batch_size=512 | tee -a $file
	echo "sdvrp beam_size = 50" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/vrp_50.pkl --model=pretrained/sdvrp_50/epoch-99.pt --beam_size=50 --eval_batch_size=512 | tee -a $file

	echo "sdvrp beam_size = 1" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/vrp_100.pkl --model=pretrained/sdvrp_100/epoch-99.pt --beam_size=1 --eval_batch_size=1024 | tee -a $file
	echo "sdvrp beam_size = 30" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/vrp_100.pkl --model=pretrained/sdvrp_100/epoch-99.pt --beam_size=30 --eval_batch_size=512 | tee -a $file
	echo "sdvrp beam_size = 50" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/vrp_100.pkl --model=pretrained/sdvrp_100/epoch-99.pt --beam_size=50 --eval_batch_size=256 | tee -a $file
fi
# ------------------------ op -----------------------
if [ "$1" = op ]
then
	echo "op beam_size = 1" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/op_20.pkl --model=pretrained/op_20/epoch-99.pt --beam_size=1 --eval_batch_size=1024 | tee -a $file
	echo "op beam_size = 30" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/op_20.pkl --model=pretrained/op_20/epoch-99.pt --beam_size=30 --eval_batch_size=1024 | tee -a $file
	echo "op beam_size = 50" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/op_20.pkl --model=pretrained/op_20/epoch-99.pt --beam_size=50 --eval_batch_size=1024 | tee -a $file

	echo "op beam_size = 1" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/op_50.pkl --model=pretrained/op_50/epoch-99.pt --beam_size=1 --eval_batch_size=1024 | tee -a $file
	echo "op beam_size = 30" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/op_50.pkl --model=pretrained/op_50/epoch-99.pt --beam_size=30 --eval_batch_size=512 | tee -a $file
	echo "op beam_size = 50" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/op_50.pkl --model=pretrained/op_50/epoch-99.pt --beam_size=50 --eval_batch_size=512 | tee -a $file

	echo "op beam_size = 1" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/op_100.pkl --model=pretrained/op_100/epoch-99.pt --beam_size=1 --eval_batch_size=1024 | tee -a $file
	echo "op beam_size = 30" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/op_100.pkl --model=pretrained/op_100/epoch-99.pt --beam_size=30 --eval_batch_size=512 | tee -a $file
	echo "op beam_size = 50" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/op_100.pkl --model=pretrained/op_100/epoch-99.pt --beam_size=50 --eval_batch_size=256 | tee -a $file
fi
# ------------------------ pctsp -----------------------
if [ "$1" = pctsp ]
then
	echo "pctsp beam_size = 1" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/pctsp_20.pkl --model=pretrained/pctsp_20/epoch-99.pt --beam_size=1 --eval_batch_size=1024 | tee -a $file
	echo "pctsp beam_size = 30" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/pctsp_20.pkl --model=pretrained/pctsp_20/epoch-99.pt --beam_size=30 --eval_batch_size=1024 | tee -a $file
	echo "pctsp beam_size = 50" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/pctsp_20.pkl --model=pretrained/pctsp_20/epoch-99.pt --beam_size=50 --eval_batch_size=1024 | tee -a $file

	echo "pctsp beam_size = 1" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/pctsp_50.pkl --model=pretrained/pctsp_50/epoch-99.pt --beam_size=1 --eval_batch_size=1024 | tee -a $file
	echo "pctsp beam_size = 30" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/pctsp_50.pkl --model=pretrained/pctsp_50/epoch-99.pt --beam_size=30 --eval_batch_size=512 | tee -a $file
	echo "pctsp beam_size = 50" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/pctsp_50.pkl --model=pretrained/pctsp_50/epoch-99.pt --beam_size=50 --eval_batch_size=512 | tee -a $file

	echo "pctsp beam_size = 1" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/pctsp_100.pkl --model=pretrained/pctsp_100/epoch-99.pt --beam_size=1 --eval_batch_size=1024 | tee -a $file
	echo "pctsp beam_size = 30" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/pctsp_100.pkl --model=pretrained/pctsp_100/epoch-99.pt --beam_size=30 --eval_batch_size=512 | tee -a $file
	echo "pctsp beam_size = 50" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/pctsp_100.pkl --model=pretrained/pctsp_100/epoch-99.pt --beam_size=50 --eval_batch_size=256 | tee -a $file
fi
# ------------------------ pctsp -----------------------
if [ "$1" = spctsp ]
then
	echo "cspctsp beam_size = 1" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/pctsp_20.pkl --model=pretrained/spctsp_20/epoch-99.pt --beam_size=1 --eval_batch_size=1024 | tee -a $file
	echo "cspctsp beam_size = 30" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/pctsp_20.pkl --model=pretrained/spctsp_20/epoch-99.pt --beam_size=30 --eval_batch_size=1024 | tee -a $file
	echo "cspctsp beam_size = 50" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/pctsp_20.pkl --model=pretrained/spctsp_20/epoch-99.pt --beam_size=50 --eval_batch_size=1024 | tee -a $file

	echo "cspctsp beam_size = 1" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/pctsp_50.pkl --model=pretrained/spctsp_50/epoch-99.pt --beam_size=1 --eval_batch_size=1024 | tee -a $file
	echo "cspctsp beam_size = 30" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/pctsp_50.pkl --model=pretrained/spctsp_50/epoch-99.pt --beam_size=30 --eval_batch_size=512 | tee -a $file
	echo "cspctsp beam_size = 50" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/pctsp_50.pkl --model=pretrained/spctsp_50/epoch-99.pt --beam_size=50 --eval_batch_size=512 | tee -a $file

	echo "cspctsp beam_size = 1" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/pctsp_100.pkl --model=pretrained/spctsp_100/epoch-99.pt --beam_size=1 --eval_batch_size=1024 | tee -a $file
	echo "cspctsp beam_size = 30" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/pctsp_100.pkl --model=pretrained/spctsp_100/epoch-99.pt --beam_size=30 --eval_batch_size=512 | tee -a $file
	echo "cspctsp beam_size = 50" | tee -a $file
	CUDA_VISIBLE_DEVICES="0" python -u search.py data/pctsp_100.pkl --model=pretrained/spctsp_100/epoch-99.pt --beam_size=50 --eval_batch_size=256 | tee -a $file
fi
