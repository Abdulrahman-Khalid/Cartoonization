run-violajones:
	python3.7 run-live.py violajones

run-dlib:
	python3.7 run-live.py dlib

run-tests:
	python3.7 run-tests.py

relearn:
	python3.7 relearn.py

install-requirements:
	python3 -m pip install --user --verbose --timeout 180 -r requirements.txt

install-dlib:
	wget http://dlib.net/files/dlib-19.18.tar.bz2

	tar xjf dlib-19.18.tar.bz2
	cd dlib-19.18

	sed -i 's/set(USE_SSE4_INSTRUCTIONS ON CACHE BOOL "Compile your program with SSE4 instructions")/set(USE_SSE4_INSTRUCTIONS OFF CACHE BOOL "Compile your program with SSE4 instructions")/g' dlib/cmake_utils/set_compiler_specific_options.cmake dlib/cmake_utils/test_for_sse4/CMakeLists.txt
	python3.7 setup.py install --user

	cd ..
	rm -rf dlib-19.18
