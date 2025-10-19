object=rz_approx_lib
LIBS_DIR=../../../../libs

all: ${object}.dll

clean:
	del /q ${object}.dll

%.dll: %.o
	g++ -shared -I$(LIBS_DIR) -fPIC -static -o $@ $< -lstdc++

%.o: %.cpp
	g++ -c -I$(LIBS_DIR) -fPIC -o $@ $< -lstdc++
