.PHONY: cythonize

cythonize: g_math.so g_output.so physiology.so run_faster.so

%.cpp: %.spyx
	cython --cplus $<

%.so: %.cpp
	g++ -shared -o $@ -fPIC `pkg-config --cflags --libs python` $<
