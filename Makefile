algo=dlib

run:
	python3.7 main.py --model model/shape_predictor.dat --algo ${algo}

init:
	@if ! [ `command -v  python3.7` ]; then\
		sudo apt-get update && sudo apt-get install -y  python3.7;\
	fi

	@echo install python requirements
	pip3.7 install --user --verbose --timeout 180 \
					opencv-python imutils datetime argparse dlib PyQt5

dlib:
	wget http://dlib.net/files/dlib-19.18.tar.bz2

	tar xjf dlib-19.18.tar.bz2
	cd dlib-19.18

	sed -i 's/set(USE_SSE4_INSTRUCTIONS ON CACHE BOOL "Compile your program with SSE4 instructions")/set(USE_SSE4_INSTRUCTIONS OFF CACHE BOOL "Compile your program with SSE4 instructions")/g' dlib/cmake_utils/set_compiler_specific_options.cmake dlib/cmake_utils/test_for_sse4/CMakeLists.txt
	python3.7 setup.py install --user

	cd ..
	rm -rf dlib-19.18