.PHONY: cythonize

cythonize: g_math.so g_output.so physiology.so run_faster.so

numpy_dir=$(shell echo -e 'import numpy\nprint(numpy.__path__[0])' | python)

%.cpp: %.spyx
	cython -3 --cplus $<

%.so: %.cpp
	g++ -shared -o $@ -fPIC $$(pkg-config --cflags --libs python-3.10) -I"$(numpy_dir)/core/include" $(CFLAGS) $(LDFLAGS) $<
