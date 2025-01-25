object=diophantine_equation_lib

all: ${object}.dll

clean:
	del /q ${object}.dll

%.dll: %.o
	g++ -shared -fPIC -static -o $@ $< -lstdc++

%.o: %.cpp
	g++ -c -fPIC -o $@ $< -lstdc++
